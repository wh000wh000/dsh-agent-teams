/**
 * Boundary translation for the Jev decision layer.
 *
 * Jev's own documentation is explicit that English is the language it is
 * strongest in: CJK payloads are processed, but they are not equivalent, and
 * the difference shows up as probability mass that cannot be told apart. Since
 * an unreadable distribution is worse than no decision at all, the decision
 * layer never sends native-language prose — it either sends English or
 * abstains.
 *
 * This module is the "send English" half. It owns one bounded, temperature-0
 * model call that rewrites ONLY the allowlisted prose slots (`JEV_PROSE_PATHS`)
 * and returns a strict key→English map. It never returns a partial result:
 * a missing key is a failed translation, because substituting the original for
 * a fraction of the evidence would silently mix languages inside one request.
 *
 * The team's own durable state is untouched. Translation happens on the way
 * out, so members and users keep reading the language they wrote, and a
 * translation failure degrades to the pure heuristics rather than to a
 * half-translated request.
 *
 * @module dsh-agent-teams/jev-translate
 */

import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import { proseSlotKey, type JevProseSlot, type JevTextTranslator } from './jev.ts'

/** Total character budget for one translation request. */
export const MAX_TRANSLATION_CHARS = 24_000
/** Output cap for the translation call; the payload is re-emitted, not summarized. */
export const TRANSLATION_MAX_TOKENS = 4_096

/** A resolved route for the translation call. */
export interface JevTranslationRoute {
  provider: string
  model: string
}

/** Fixed instruction for the boundary translator. */
export const TRANSLATION_POLICY = [
  'You translate text into English for a downstream classifier.',
  'You receive a JSON object whose keys identify fields and whose values are the text to translate.',
  'Return ONE JSON object with exactly the same keys, in the same order, whose values are the English translation of each input value.',
  'Preserve identifiers, code identifiers, file paths, command lines, error strings, and numbers exactly as written; translate only the surrounding prose.',
  'Do not add fields, drop fields, merge fields, explain, or wrap the result in prose or code fences.',
  'If a value is already English, return it unchanged.',
].join(' ')

/**
 * Extract the one JSON object from a model reply, tolerating a single code
 * fence. Anything else is a failed translation.
 */
export function parseTranslationObject(text: string): Record<string, string> | undefined {
  const trimmed = text.trim()
  const fenced = /^```(?:json)?\s*\n([\s\S]*?)\n?```$/u.exec(trimmed)
  const body = (fenced?.[1] ?? trimmed).trim()
  if (body === '') return undefined
  let value: unknown
  try {
    value = JSON.parse(body)
  } catch {
    return undefined
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return undefined
  const record: Record<string, string> = {}
  for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
    if (typeof item !== 'string' || item === '') return undefined
    record[key] = item
  }
  return record
}

/**
 * Validate a parsed reply against the requested slots.
 *
 * Every requested key must be present exactly once with a non-empty string;
 * a partial reply is rejected whole so no request ever mixes languages.
 */
export function completeTranslation(
  requested: readonly JevProseSlot[],
  record: Readonly<Record<string, string>>,
): Map<string, string> | undefined {
  const resolved = new Map<string, string>()
  for (const slot of requested) {
    const value = record[proseSlotKey(slot)]
    if (value === undefined || value === '') return undefined
    resolved.set(proseSlotKey(slot), value)
  }
  return resolved
}

/**
 * Create the LLM-backed boundary translator.
 *
 * @param ctx - injected plugin context carrying the `llm` service.
 * @param route - resolved translation route, or `undefined` when no member of
 *   the team exposes one. An unresolved route makes the translator report
 *   itself unavailable, which the decision layer turns into an abstention
 *   instead of a native-language request.
 * @param onDiagnostic - receives one short line per failed call; never content.
 */
export function createLlmTranslator(
  ctx: Context,
  route: JevTranslationRoute | undefined,
  onDiagnostic: (message: string) => void = () => {},
): JevTextTranslator {
  if (route === undefined || route.provider === '' || route.model === '') {
    return { available: false, translate: () => Promise.resolve(undefined) }
  }

  return {
    available: true,
    async translate(slots: readonly JevProseSlot[], signal: AbortSignal): Promise<Map<string, string> | undefined> {
      if (slots.length === 0) return new Map()
      const payload: Record<string, string> = {}
      let budget = 0
      for (const slot of slots) {
        const key = proseSlotKey(slot)
        if (payload[key] !== undefined) continue
        budget += slot.value.length
        if (budget > MAX_TRANSLATION_CHARS) {
          onDiagnostic('agent-teams: boundary translation skipped because the review exceeds the translation budget')
          return undefined
        }
        payload[key] = slot.value
      }

      try {
        const assembler = new BlockAssembler()
        let finished = false
        const stream = ctx.llm.stream({
          provider: route.provider,
          model: route.model,
          system: TRANSLATION_POLICY,
          messages: [createUserMessage({
            content: [{ type: 'text', text: JSON.stringify(payload) }],
            source: { kind: 'plugin', plugin: 'dsh-agent-teams' },
          })],
          temperature: 0,
          maxTokens: TRANSLATION_MAX_TOKENS,
          signal,
        })
        for await (const chunk of stream) {
          assembler.push(chunk)
          if (chunk.type === 'finish') {
            finished = true
            if (chunk.reason.kind !== 'stop') {
              onDiagnostic(`agent-teams: boundary translation ended with ${chunk.reason.kind}`)
              return undefined
            }
          }
        }
        if (!finished) {
          onDiagnostic('agent-teams: boundary translation emitted no terminal finish')
          return undefined
        }
        const text = assembler.blocks()
          .filter((block) => block.type === 'text')
          .map((block) => block.text)
          .join('')
        const parsed = parseTranslationObject(text)
        if (parsed === undefined) {
          onDiagnostic('agent-teams: boundary translation did not return one JSON object')
          return undefined
        }
        const complete = completeTranslation(slots, parsed)
        if (complete === undefined) {
          onDiagnostic('agent-teams: boundary translation returned an incomplete key set')
          return undefined
        }
        return complete
      } catch (error: unknown) {
        const reason = error instanceof Error ? error.name : 'error'
        onDiagnostic(`agent-teams: boundary translation failed (${reason})`)
        return undefined
      }
    },
  }
}

/**
 * Choose the translation route from the team's own durable records.
 *
 * The reviewing member's captured route is the natural choice: it is already
 * validated, already paid for, and it is the model that produced the prose
 * being translated. No route is invented and none is silently defaulted,
 * because a translation routed to an unavailable model would turn one
 * abstention into a permanent outage of the decision layer.
 */
export function translationRouteFromTeam(
  members: readonly { status?: string, provider?: string, model?: string, activeProvider?: string, activeModel?: string, name: string }[],
  preferredNames: readonly (string | undefined)[],
): JevTranslationRoute | undefined {
  const live = members.filter((member) => member.status !== 'removed')
  const ordered = [
    ...live.filter((member) => preferredNames.includes(member.name)),
    ...live,
  ]
  for (const member of ordered) {
    const provider = (member.activeProvider ?? member.provider ?? '').trim()
    const model = (member.activeModel ?? member.model ?? '').trim()
    if (provider !== '' && model !== '') return { provider, model }
  }
  return undefined
}