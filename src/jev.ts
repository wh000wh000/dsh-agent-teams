/**
 * Jev decision layer for AgentTeams quality gates.
 *
 * Jev (TypeSafe System One Model) answers a fixed, pre-enumerated set of
 * questions about unstructured `state` in one forward pass and returns TYPED
 * PROBABILITY DISTRIBUTIONS — it never generates text. That shape is exactly
 * what the three heuristics in `quality-gates.ts` lack: each of them is a
 * judgement (which files must change, who should own the work, whether a
 * finding restates an earlier one) currently hard-coded as a regex, a
 * positional fallback, or an id-keyed string comparison.
 *
 * ## Division of responsibility
 *
 * This module owns ALL of the I/O and none of the decision semantics:
 *
 * - `quality-gates.ts` stays pure and authoritative. The hints this module
 *   returns are advisory: every one of them can be discarded by the pure
 *   layer, which then falls back to the heuristic it always used.
 * - This module owns the network call, the timeout, the abstention rule, and
 *   the threshold. Every failure path returns `{}` — the layer is fail-open by
 *   construction, because a decision service that can block a repair loop is
 *   worse than no decision service.
 *
 * ## Three rules this implementation obeys deliberately
 *
 * 1. **Abstention uses `probabilities`, never `confidence`.** `confidence` is
 *    a summary statistic derived from the distribution, not the probability
 *    that the answer is right. The official abstention rule keys on the top
 *    probability of the distribution, and the threshold must be calibrated on
 *    the caller's own labelled data — the default here is the published
 *    starting point, not a claim.
 * 2. **Questions are isolated from each other.** A question cannot read a
 *    sibling question's content or answer, so every decision carries its own
 *    evidence in the shared `state` and nothing is chained inside one call.
 * 3. **Every enumerated answer set has an explicit abstention exit**, and the
 *    caller must route it somewhere (here: back to the heuristic).
 *
 * @module dsh-agent-teams/jev
 */

import type { ReviewFinding, TeamState, TeamTask } from './types.ts'
import {
  DEFAULT_KEYCHAIN_ACCOUNT,
  DEFAULT_KEYCHAIN_SERVICE,
  describeCredentialOrigin,
  resolveJevCredential,
  type JevCredential,
} from './jev-credential.ts'
import type { JevDecisionHints } from './quality-gates.ts'
import { repairScopeFromFindings } from './quality-gates.ts'

/** Environment variable holding the API key when the config names no other. */
export const DEFAULT_JEV_API_KEY_ENV = 'JEV_API_KEY'
/** Public System One endpoint. */
export const DEFAULT_JEV_BASE_URL = 'https://api.typesafe.ai'
/**
 * Published abstention starting point: below this top probability the model is
 * treated as unable to distinguish the options and the answer is discarded.
 * Calibrate this on your own labelled decisions before trusting it.
 */
export const DEFAULT_JEV_MIN_PROBABILITY = 0.6
/** Hard cap on one request's question count; the model accepts far more, but a review round never needs it. */
export const MAX_JEV_QUESTIONS = 200
/** Cap on candidate paths considered in one repair-scope decision. */
export const MAX_SCOPE_CANDIDATES = 120

/** Resolved decision-layer configuration. */
export interface JevDecisionConfig {
  /** Master switch. When false the layer is never invoked. */
  enabled: boolean
  /** System One endpoint, without a trailing slash. */
  baseUrl: string
  /** Pinned model id. Never use a moving alias: thresholds silently rot. */
  model: string
  /** Name of the environment variable holding the API key. */
  apiKeyEnv: string
  /** Keychain service holding the API key when no environment variable does. */
  keychainService: string
  /** Keychain account holding the API key when no environment variable does. */
  keychainAccount: string
  /** End-to-end timeout in milliseconds; expiry is a fail-open. */
  timeoutMs: number
  /**
   * Global top-probability floor, used by any decision that has no explicit
   * override of its own.
   */
  minProbability: number
  /** Individual decisions that may be delegated. */
  decisions: {
    repairScope: boolean
    routing: boolean
    dedup: boolean
    /**
     * How a repair-scope answer combines with the existing derivation.
     * `union` keeps every path the findings observe or name and lets the
     * decision add to that set; `replace` takes the answer verbatim. See
     * `JevDecisionHints.scopePolicy`.
     */
    scopePolicy: 'union' | 'replace'
    /**
     * Per-decision floors. A threshold is a property of ONE decision and ONE
     * labelled corpus, so a single global number silently applies a
     * calibration to decisions it was never measured on. Overrides exist so a
     * calibration can be introduced where it was actually measured.
     */
    minProbability: {
      repairScope: number
      routing: number
      dedup: number
    }
  }
}

/** Raw plugin config shape (all fields optional). */
export interface JevConfigInput {
  enabled?: boolean
  baseUrl?: string
  model?: string
  apiKeyEnv?: string
  keychainService?: string
  keychainAccount?: string
  timeoutMs?: number
  minProbability?: number
  decisions?: {
    repairScope?: boolean
    routing?: boolean
    dedup?: boolean
    scopePolicy?: 'union' | 'replace'
    minProbability?: {
      repairScope?: number
      routing?: number
      dedup?: number
    }
  }
}

/** Resolve a plugin config block into a complete decision-layer config. */
export function resolveJevConfig(input: JevConfigInput | undefined): JevDecisionConfig {
  return {
    enabled: input?.enabled === true,
    baseUrl: (input?.baseUrl ?? DEFAULT_JEV_BASE_URL).replace(/\/+$/u, ''),
    model: input?.model ?? 'jev-latest',
    apiKeyEnv: input?.apiKeyEnv ?? DEFAULT_JEV_API_KEY_ENV,
    keychainService: input?.keychainService ?? DEFAULT_KEYCHAIN_SERVICE,
    keychainAccount: input?.keychainAccount ?? DEFAULT_KEYCHAIN_ACCOUNT,
    timeoutMs: input?.timeoutMs ?? 4_000,
    minProbability: input?.minProbability ?? DEFAULT_JEV_MIN_PROBABILITY,
    decisions: {
      repairScope: input?.decisions?.repairScope ?? true,
      routing: input?.decisions?.routing ?? true,
      dedup: input?.decisions?.dedup ?? true,
      scopePolicy: input?.decisions?.scopePolicy ?? 'union',
      minProbability: {
        repairScope: input?.decisions?.minProbability?.repairScope ?? input?.minProbability ?? DEFAULT_JEV_MIN_PROBABILITY,
        routing: input?.decisions?.minProbability?.routing ?? input?.minProbability ?? DEFAULT_JEV_MIN_PROBABILITY,
        dedup: input?.decisions?.minProbability?.dedup ?? input?.minProbability ?? DEFAULT_JEV_MIN_PROBABILITY,
      },
    },
  }
}

/** One question as sent to System One. */
export interface JevQuestion {
  type: 'choice' | 'noul'
  instructions: string
  criteria?: Record<string, string>
}

/** One answer as returned by System One. */
export interface JevAnswer {
  type: 'choice' | 'noul' | 'score'
  choice?: string
  noul?: number
  confidence?: number
  probabilities?: Record<string, number>
}

/** Transport seam so tests never touch the network. */
export type JevFetch = (input: string, init: RequestInit) => Promise<{
  ok: boolean
  status: number
  json: () => Promise<unknown>
}>

/** Everything this layer needs about one failed review. */
export interface JevDecisionInput {
  team: TeamState
  closed: TeamTask
  /** Candidate workspace-relative paths for the repair scope; the caller supplies them from the real workspace. */
  scopeCandidates?: readonly string[]
}

/**
 * The prose paths inside the request `state` that may need an English
 * projection, as `path[*]`-style keys (`*` expands per array element).
 *
 * This list is deliberately an ALLOWLIST. Paths, ids, kinds, statuses,
 * verdicts, severities, member names and rounds are language-neutral by
 * construction, so they are never handed to a translator — a translator that
 * rewrites `server/src/routes/refund.ts` would break the whole decision.
 */
export const JEV_PROSE_PATHS = [
  'review.objective',
  'review.acceptance[*]',
  'findings[*].problem',
  'findings[*].requiredFix',
  'earlierFindings[*].problem',
  'earlierFindings[*].requiredFix',
  'roster[*].description',
] as const

/** One prose slot collected from the request state. */
export interface JevProseSlot {
  /** Stable allowlist key identifying where the value came from. */
  path: string
  /** Index of the array element for `[*]` paths, absent for scalar paths. */
  index?: number
  /** The original, native-language value. */
  value: string
}

/** True when a string carries characters outside the Latin/ASCII range. */
export function needsEnglishProjection(value: string): boolean {
  return /[^\u0000-\u007F]/u.test(value)
}

/** Whether any collected prose slot needs translation. */
export function requiresTranslation(slots: readonly JevProseSlot[]): boolean {
  return slots.some((slot) => needsEnglishProjection(slot.value))
}

/**
 * A boundary translator: turns the collected native-language prose into
 * English before it reaches Jev.
 *
 * DSH's own documentation states English is the model's strongest language;
 * CJK is processed but is not equivalent, which is why this seam exists and
 * why the layer abstains rather than sending untranslated prose.
 */
export interface JevTextTranslator {
  /** Whether a translation attempt is possible at all (a route must resolve). */
  readonly available: boolean
  /**
   * Translate the supplied slots.
   * @returns the English values keyed by the same path and index, or
   *   `undefined` when the translation could not be produced. A partial
   *   result is a failure: the caller must not guess the missing half.
   */
  translate: (slots: readonly JevProseSlot[], signal: AbortSignal) => Promise<Map<string, string> | undefined>
}

/** Stable map key for one prose slot. */
export function proseSlotKey(slot: Pick<JevProseSlot, 'path' | 'index'>): string {
  return slot.index === undefined ? slot.path : `${slot.path}#${String(slot.index)}`
}

/** Apply a translated value map back onto the collected slots. */
export function applyTranslation(
  slots: readonly JevProseSlot[],
  translated: ReadonlyMap<string, string>,
): Map<string, string> {
  const resolved = new Map<string, string>()
  for (const slot of slots) {
    const english = translated.get(proseSlotKey(slot))
    resolved.set(proseSlotKey(slot), english ?? slot.value)
  }
  return resolved
}

/**
 * Highest probability in one answer's distribution.
 *
 * Uses `probabilities` by design. For a `choice` answer the distribution is
 * the whole option set; for a `noul` answer the returned `noul` value IS the
 * probability of yes, so its distance from an even split is what carries
 * information — reported as the larger of yes/no.
 */
export function topProbability(answer: JevAnswer | undefined): number {
  if (answer === undefined) return 0
  if (answer.type === 'noul') {
    const yes = typeof answer.noul === 'number' ? answer.noul : 0
    return Math.max(yes, 1 - yes)
  }
  const values = Object.values(answer.probabilities ?? {})
  return values.length === 0 ? 0 : Math.max(...values)
}

/**
 * Accept a choice answer only when it clears the abstention floor and names an
 * option the caller enumerated.
 *
 * Returns `undefined` for every rejection path — abstained option, unknown
 * option, missing answer, or a distribution the model could not separate —
 * so the caller falls back to its heuristic rather than acting on a guess.
 */
export function acceptedChoice(
  answer: JevAnswer | undefined,
  options: readonly string[],
  minProbability: number,
): string | undefined {
  if (answer?.type !== 'choice') return undefined
  const choice = answer.choice
  if (choice === undefined || !options.includes(choice)) return undefined
  if (choice === 'unknown') return undefined
  if (topProbability(answer) < minProbability) return undefined
  return choice
}

/**
 * Accept a Noul answer as a boolean only when the distribution clears the
 * floor. A 0.5 answer means the model cannot separate yes from no — it is not
 * a medium value, and it must not be read as agreement.
 */
export function acceptedNoul(
  answer: JevAnswer | undefined,
  minProbability: number,
): boolean | undefined {
  if (answer?.type !== 'noul') return undefined
  const yes = answer.noul
  if (typeof yes !== 'number') return undefined
  const certainty = Math.max(yes, 1 - yes)
  if (certainty < minProbability) return undefined
  return yes >= 0.5
}

/**
 * Build the repair-scope questions: one three-way choice per candidate path.
 *
 * `Extraction-as-choice`: the candidate set comes from the caller's real
 * workspace (changed files, declared scope, paths named by the findings), and
 * the model may only PICK among existing paths. It never produces a path, so
 * the failure mode of text extraction — a token that is not a path, or a path
 * mangled by surrounding punctuation — cannot happen by construction.
 */
export function buildScopeQuestions(
  findingIds: readonly string[],
  candidates: readonly string[],
): Record<string, JevQuestion> {
  const criteria = {
    include: 'Resolving this finding requires editing this path',
    exclude: 'This path is context, a citation, or unrelated to the fix',
    unknown: 'The finding does not give enough evidence to decide',
  }
  const questions: Record<string, JevQuestion> = {}
  for (const findingId of findingIds) {
    for (const path of candidates) {
      questions[`scope::${findingId}::${path}`] = {
        type: 'choice',
        instructions: [
          `Consider only finding "${findingId}" and the repository state.`,
          `Must \`${path}\` be modified to resolve this finding?`,
          'Answer include only when the fix the finding requires edits this path.',
          'Answer exclude when the path is merely cited, is context, or is a plausible file the finding never makes necessary.',
          'Answer unknown when the finding does not say enough to decide.',
        ].join(' '),
        criteria,
      }
    }
  }
  return questions
}

/**
 * Build the ownership questions: one choice per generated task.
 *
 * The option set is the live roster, so the model cannot invent a member. The
 * forbidden member is stated in the instructions AND filtered again by the
 * pure layer, because a hint is advisory and must not widen who is
 * schedulable.
 */
export function buildRoutingQuestions(
  tasks: readonly { id: string, kind: string, objective: string, notes?: string }[],
  roster: readonly { name: string, role?: string, description?: string }[],
): Record<string, JevQuestion> {
  const criteria: Record<string, string> = {}
  for (const member of roster) {
    criteria[member.name] = [member.role ?? 'member', member.description ?? ''].filter(Boolean).join(': ')
  }
  criteria['unknown'] = 'No roster member matches this work'
  const questions: Record<string, JevQuestion> = {}
  for (const task of tasks) {
    questions[`owner::${task.id}`] = {
      type: 'choice',
      instructions: [
        `Decide the owner of task ${task.id} (kind=${task.kind}, objective="${task.objective}").`,
        task.notes ?? '',
        'Pick the member whose role and description match the actual work.',
        'Choose unknown when no member is a real match; abstaining is preferred over assigning work to the wrong role.',
      ].filter((line) => line !== '').join(' '),
      criteria,
    }
  }
  return questions
}

/**
 * Build the duplicate-defect questions: one Noul per (incoming, earlier) pair.
 *
 * This is what keeps the repair-attempt budget anchored on the DEFECT instead
 * of on the identifier a reviewer happened to choose this round.
 */
export function buildDedupQuestions(
  incoming: readonly { id: string, problem: string, requiredFix: string }[],
  earlier: readonly { id: string, problem: string, requiredFix: string }[],
): Record<string, JevQuestion> {
  const questions: Record<string, JevQuestion> = {}
  for (const next of incoming) {
    for (const previous of earlier) {
      questions[`dup::${next.id}::${previous.id}`] = {
        type: 'noul',
        instructions: [
          `Does newly reported finding ${next.id} describe the same underlying defect as the earlier finding ${previous.id}?`,
          'Answer yes when fixing the earlier finding already fixes, or necessarily fixes, this one — even though the wording, identifier, and line numbers differ.',
          'Answer no when they are independent defects that can each exist on their own.',
        ].join(' '),
      }
    }
  }
  return questions
}

/**
 * Earlier findings worth comparing against when deciding whether a new finding
 * restates a defect already reported.
 *
 * Scope matters more than it looks. A first cut compared against findings on
 * EVERY task, which on a real team pulled 22 unrelated findings out of two
 * `work` tasks into the comparison: it diluted the question, and because the
 * boundary translator has to render every slot it also inflated the request.
 *
 * The comparison set is now the closed review's own LINEAGE: walk
 * `reviewedTaskId` / `sourceTaskId` from the closed task to its root, then
 * include findings from any task on that chain, plus any task that points at a
 * chain member. That covers the round-to-round case a repair budget depends on
 * — review r2 points at the repair, the repair points at the original
 * implementation, and the r1 review points at that same implementation — while
 * excluding findings from unrelated work.
 */
export function earlierFindings(
  team: TeamState,
  closed: { id: string, reviewedTaskId?: string, sourceTaskId?: string },
  incomingIds: readonly string[],
): { id: string, problem: string, requiredFix: string }[] {
  const lineage = new Set<string>()
  const pending: (string | undefined)[] = [closed.id, closed.reviewedTaskId, closed.sourceTaskId]
  while (pending.length > 0) {
    const id = pending.pop()
    if (id === undefined || id === '' || lineage.has(id)) continue
    lineage.add(id)
    const task = team.tasks.find((candidate) => candidate.id === id)
    if (task === undefined) continue
    pending.push(task.reviewedTaskId, task.sourceTaskId)
  }

  const seen = new Set<string>()
  const result: { id: string, problem: string, requiredFix: string }[] = []
  for (const task of team.tasks) {
    const related = lineage.has(task.id)
      || (task.reviewedTaskId !== undefined && lineage.has(task.reviewedTaskId))
      || (task.sourceTaskId !== undefined && lineage.has(task.sourceTaskId))
    if (!related) continue
    for (const finding of task.findings ?? []) {
      if (incomingIds.includes(finding.id) || seen.has(finding.id)) continue
      seen.add(finding.id)
      result.push({ id: finding.id, problem: finding.problem, requiredFix: finding.requiredFix })
    }
  }
  return result
}

/**
 * Fold one System One response into the hint set the pure layer accepts.
 *
 * Exported for tests: it is the whole strategy layer (threshold, abstention,
 * option validation) with the network removed.
 */
export function hintsFromAnswers(
  answers: Record<string, JevAnswer> | undefined,
  input: {
    findingIds: readonly string[]
    scopeCandidates: readonly string[]
    roster: readonly string[]
    routingTasks: readonly string[]
    earlierFindingIds: readonly string[]
    decisions: JevDecisionConfig['decisions']
  },
): JevDecisionHints {
  const hints: JevDecisionHints = {}
  const map = answers ?? {}

  if (input.decisions.repairScope) {
    const scope: string[] = []
    for (const findingId of input.findingIds) {
      for (const path of input.scopeCandidates) {
        const choice = acceptedChoice(map[`scope::${findingId}::${path}`], ['include', 'exclude', 'unknown'], input.decisions.minProbability.repairScope)
        if (choice === 'include' && !scope.includes(path)) scope.push(path)
      }
    }
    if (scope.length > 0) {
      hints.repairInScope = scope
      hints.scopePolicy = input.decisions.scopePolicy
    }
  }

  if (input.decisions.routing) {
    for (const taskId of input.routingTasks) {
      const choice = acceptedChoice(map[`owner::${taskId}`], [...input.roster, 'unknown'], input.decisions.minProbability.routing)
      if (choice === undefined) continue
      if (taskId === 'repair') hints.implementer = choice
      if (taskId === 'review') hints.reviewer = choice
    }
  }

  if (input.decisions.dedup) {
    const aliases: Record<string, string[]> = {}
    for (const findingId of input.findingIds) {
      // Collect EVERY earlier finding the model confirmed, not just the most
      // confident one. A review may merge several earlier findings into a
      // single narrative, and keeping only the best match would lose the rest
      // of the merged set — which is exactly the case that used to leave the
      // repair budget uncounted.
      const confirmed: string[] = []
      for (const earlierId of input.earlierFindingIds) {
        if (acceptedNoul(map[`dup::${findingId}::${earlierId}`], input.decisions.minProbability.dedup) !== true) continue
        confirmed.push(earlierId)
      }
      if (confirmed.length > 0) aliases[findingId] = confirmed
    }
    if (Object.keys(aliases).length > 0) hints.findingAliases = aliases
  }

  return hints
}

/** Interface implemented by the live client; a no-op implementation is used when the layer is off. */
export interface JevDecisions {
  /** Whether a real decision call would be attempted for this input. */
  readonly enabled: boolean
  /** Resolve hints for one failed review. Never throws; returns `{}` on every failure path. */
  decide: (input: JevDecisionInput) => Promise<JevDecisionHints>
}

/** A permanently disabled decision layer. */
export function disabledJevDecisions(): JevDecisions {
  return {
    enabled: false,
    decide: () => Promise.resolve({}),
  }
}

/** Injectable seams for the live decision layer. */
export interface JevDecisionDeps {
  /** Transport override; defaults to global `fetch`. */
  fetch?: JevFetch
  /**
   * Boundary translator, or a resolver called once per decision with that
   * decision's input. The resolver form exists because the natural route — the
   * reviewing member's own captured model — is only known once a team and a
   * failed review exist, not at plugin mount time.
   */
  translator?: JevTextTranslator | ((input: JevDecisionInput) => JevTextTranslator | undefined)
  /**
   * Credential resolver override. Defaults to environment-then-keychain
   * resolution; injected so tests never read a real secret.
   */
  credential?: () => Promise<JevCredential | undefined>
  /** Receives one short line per failed or skipped call; never receives the credential. */
  onDiagnostic?: (message: string) => void
}

/** Resolve the translator for one decision. */
function translatorFor(
  translator: JevDecisionDeps['translator'],
  input: JevDecisionInput,
): JevTextTranslator | undefined {
  return typeof translator === 'function' ? translator(input) : translator
}

/**
 * Collect every prose slot an English projection would have to rewrite.
 *
 * Only allowlisted fields are collected: a translator is never pointed at a
 * path, an identifier, or any other value whose exact bytes matter.
 */
export function collectProseSlots(
  input: JevDecisionInput,
  findings: readonly ReviewFinding[],
  earlier: readonly { id: string, problem: string, requiredFix: string }[],
  roster: readonly { name: string, role?: string, description?: string }[],
): JevProseSlot[] {
  const slots: JevProseSlot[] = []
  const push = (path: string, index: number | undefined, value: string | undefined): void => {
    if (value === undefined || value === '') return
    slots.push(index === undefined ? { path, value } : { path, index, value })
  }
  push('review.objective', undefined, input.closed.objective)
  ;(input.closed.acceptance ?? []).forEach((item, index) => { push('review.acceptance[*]', index, item) })
  findings.forEach((finding, index) => {
    push('findings[*].problem', index, finding.problem)
    push('findings[*].requiredFix', index, finding.requiredFix)
  })
  earlier.forEach((finding, index) => {
    push('earlierFindings[*].problem', index, finding.problem)
    push('earlierFindings[*].requiredFix', index, finding.requiredFix)
  })
  roster.forEach((member, index) => { push('roster[*].description', index, member.description) })
  return slots
}

/**
 * Build the request `state`, substituting the English projection where one
 * was produced. Language-neutral fields are copied verbatim by construction.
 */
export function projectState(
  input: JevDecisionInput,
  findings: readonly ReviewFinding[],
  earlier: readonly { id: string, problem: string, requiredFix: string }[],
  roster: readonly { name: string, role?: string, description?: string }[],
  scopeCandidates: readonly string[],
  resolved: ReadonlyMap<string, string>,
): Record<string, unknown> {
  const at = (path: string, index: number | undefined, fallback: string | undefined): string | undefined => {
    const key = index === undefined ? path : `${path}#${String(index)}`
    return resolved.get(key) ?? fallback
  }
  return {
    review: {
      taskId: input.closed.id,
      kind: input.closed.kind ?? 'review',
      round: input.closed.round ?? 1,
      objective: at('review.objective', undefined, input.closed.objective),
      acceptance: (input.closed.acceptance ?? [])
        .map((item, index) => at('review.acceptance[*]', index, item)),
      verdict: input.closed.verdict,
    },
    findings: findings.map((finding, index) => ({
      id: finding.id,
      severity: finding.severity,
      file: finding.file,
      problem: at('findings[*].problem', index, finding.problem),
      requiredFix: at('findings[*].requiredFix', index, finding.requiredFix),
    })),
    earlierFindings: earlier.map((finding, index) => ({
      id: finding.id,
      problem: at('earlierFindings[*].problem', index, finding.problem),
      requiredFix: at('earlierFindings[*].requiredFix', index, finding.requiredFix),
    })),
    scopeCandidates,
    roster,
    heuristicScope: repairScopeFromFindings(findings, undefined),
  }
}

/**
 * Create the live decision layer.
 *
 * @param config - resolved decision-layer configuration.
 * @param env - environment map holding the API key. Injected so tests and the
 *   host can supply a scoped map instead of the process environment.
 * @param deps - transport, translator, and diagnostic seams.
 */
export function createJevDecisions(
  config: JevDecisionConfig,
  env: Readonly<Record<string, string | undefined>>,
  deps: JevDecisionDeps = {},
): JevDecisions {
  const { fetch: fetchImpl, onDiagnostic = () => {} } = deps
  if (!config.enabled) return disabledJevDecisions()
  const transport = fetchImpl ?? (globalThis.fetch as unknown as JevFetch)

  // The credential is resolved on first use, not at mount. Mount happens while
  // the profile is still activating, where a keychain read would be both
  // premature and unobservable; use time is where a missing credential can be
  // reported next to the decision it prevented. The attempt is cached, so a
  // failed resolution is reported once rather than on every failed review.
  let credentialAttempt: Promise<JevCredential | undefined> | undefined
  let missingReported = false
  let originReported = false
  let nativeProseReported = false
  const loadCredential = (): Promise<JevCredential | undefined> => {
    credentialAttempt ??= (deps.credential ?? (() => resolveJevCredential({
      apiKeyEnv: config.apiKeyEnv,
      env,
      service: config.keychainService,
      account: config.keychainAccount,
    })))()
    return credentialAttempt
  }

  return {
    enabled: true,
    async decide(input: JevDecisionInput): Promise<JevDecisionHints> {
      try {
        const credential = await loadCredential()
        if (credential === undefined) {
          if (!missingReported) {
            missingReported = true
            onDiagnostic(`agent-teams: Jev decisions are enabled but no credential resolved (${config.apiKeyEnv}, TYPESAFE_API_KEY, or keychain ${config.keychainService}/${config.keychainAccount}); falling back to heuristics`)
          }
          return {}
        }
        const apiKey = credential.key
        if (!originReported) {
          originReported = true
          // Names where the credential came from, never the credential.
          onDiagnostic(`agent-teams: Jev decisions using the credential from ${describeCredentialOrigin(credential.origin)}`)
        }
        const findings = (input.closed.findings ?? []).filter((finding) => finding.resolved !== true)
        const findingIds = findings.map((finding) => finding.id)
        if (findingIds.length === 0) return {}

        const scopeCandidates = [...new Set(input.scopeCandidates ?? [])]
          .slice(0, MAX_SCOPE_CANDIDATES)
        const roster = input.team.members
          .filter((member) => member.status !== 'removed' && member.name !== 'captain')
          .map((member) => ({ name: member.name, role: member.role, description: member.executionPrompt }))
        const rosterNames = roster.map((member) => member.name)
        const earlier = earlierFindings(input.team, input.closed, findingIds)
        // The two routing questions used to carry these constraints BACKWARDS:
        // the repair question forbade nothing while the review question forbade
        // the very reviewer who raised the findings. On a real team that made
        // the model assign a repair to the independent verifier (whose own
        // description says it never edits implementation code) and hand the
        // follow-up review to an auditor instead of the verifier. The intent is
        // now stated per task, and the only hard guarantee — that the reviewer
        // is not the repairer — stays enforced by the pure layer, which is the
        // layer that actually knows the resolved implementer.
        const routingTasks = [
          {
            id: 'repair',
            kind: 'repair',
            objective: input.closed.objective ?? 'Fix the review findings',
            notes: 'Prefer the member who implemented the work being repaired, when that member is still on the roster. The member who only verifies or only reviews is a poor owner for a repair, because that role does not edit implementation code.',
          },
          {
            id: 'review',
            kind: 'review',
            objective: 'Independently review the repair',
            notes: 'Choose a member who can judge the repair independently of the person who performed it. Do not choose the member who will be assigned the repair.',
          },
        ]

        const questions: Record<string, JevQuestion> = {
          ...(config.decisions.repairScope ? buildScopeQuestions(findingIds, scopeCandidates) : {}),
          ...(config.decisions.routing ? buildRoutingQuestions(routingTasks, roster) : {}),
          ...(config.decisions.dedup ? buildDedupQuestions(findings, earlier) : {}),
        }
        const ids = Object.keys(questions).slice(0, MAX_JEV_QUESTIONS)
        if (ids.length === 0) return {}
        const trimmed: Record<string, JevQuestion> = {}
        for (const id of ids) {
          const question = questions[id]
          if (question !== undefined) trimmed[id] = question
        }

        // Boundary rule: the model's strongest language is English, so the
        // request carries English prose even when the team's own artifacts are
        // not. The projection is an allowlist (see JEV_PROSE_PATHS), which is
        // why paths, ids and kinds are never rewritten.
        // Native prose goes to the model as written. A measured comparison on a
        // real team's findings found Chinese and English score identically on
        // the dedup decision (4/10 full coverage, 1 miss, 0 false merges under
        // both), so refusing to send non-English prose bought no accuracy while
        // it silently disabled the whole layer on Chinese teams. English
        // remains available as an explicit opt-in for anyone who wants it.
        const prose = collectProseSlots(input, findings, earlier, roster)
        let resolved = applyTranslation(prose, new Map())
        if (requiresTranslation(prose)) {
          const translator = translatorFor(deps.translator, input)
          if (translator?.available !== true) {
            if (!nativeProseReported) {
              nativeProseReported = true
              onDiagnostic('agent-teams: sending non-English prose to Jev as written; the model is strongest in English, so calibrate the abstention floor on your own language and data')
            }
          } else {
            const signal = AbortSignal.timeout(config.timeoutMs)
            const translated = await translator.translate(prose, signal)
            if (translated === undefined) {
              // A failed opt-in translation is not a reason to lose the
              // decision: native prose is already an accepted input.
              onDiagnostic('agent-teams: boundary translation failed; sending the original prose instead')
            } else {
              resolved = applyTranslation(prose, translated)
            }
          }
        }

        const state = projectState(input, findings, earlier, roster, scopeCandidates, resolved)

        const signal = AbortSignal.timeout(config.timeoutMs)
        const response = await transport(`${config.baseUrl}/v1/systemone`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ state, model: config.model, questions: trimmed }),
          signal,
        })
        if (!response.ok) {
          onDiagnostic(`agent-teams: Jev decision call failed with HTTP ${String(response.status)}; falling back to heuristics`)
          return {}
        }
        const body = await response.json() as { answers?: Record<string, JevAnswer> }
        return hintsFromAnswers(body.answers, {
          findingIds,
          scopeCandidates,
          roster: rosterNames,
          routingTasks: routingTasks.map((task) => task.id),
          earlierFindingIds: earlier.map((finding) => finding.id),
          decisions: config.decisions,
        })
      } catch (error: unknown) {
        const reason = error instanceof Error ? error.name : 'error'
        onDiagnostic(`agent-teams: Jev decision call unavailable (${reason}); falling back to heuristics`)
        return {}
      }
    },
  }
}