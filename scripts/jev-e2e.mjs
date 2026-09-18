#!/usr/bin/env node
/**
 * End-to-end verification of the Jev decision layer against the live API.
 *
 * Uses the BUILT lib/ exactly as the plugin does: createJevDecisions ->
 * hints -> planQualityFollowUp -> the generated repair/review pair. Nothing is
 * stubbed except the boundary translator, which is injected so this can run
 * outside a DSH host.
 *
 * Run:
 *   bash -c 'source ~/.agents/skills/jev/scripts/credentials/jev_keychain.sh; node scripts/jev-e2e.mjs'
 */
import { createJevDecisions, resolveJevConfig } from '../lib/jev.js'
import { planQualityFollowUp } from '../lib/quality-gates.js'

const key = process.env.JEV_KEY?.trim()
if (key === undefined || key === '') throw new Error('JEV_KEY missing: source the keychain helper first')

const member = (name, role, executionPrompt, extra = {}) => ({
  id: `member-${name}`, name, role, executionPrompt, joinedAt: 0, status: 'idle', ...extra,
})

const team = {
  name: 't', id: 't', description: 'Refund idempotency', captainSessionId: 'captain',
  createdAt: 0, taskSeq: 4,
  members: [
    member('impl-1', 'engineer', 'Implements TypeScript changes under server/ and runs the test suite.'),
    member('docs-1', 'writer', 'Owns docs/ and release notes; no backend access.'),
    member('qa-1', 'verifier', 'Runs verification commands and writes reproduction cases; never edits features.'),
    member('rev-1', 'reviewer', 'Reviews diffs against the contract and returns a verdict.'),
  ],
  tasks: [{
    id: 't1', subject: 'implementation', status: 'completed', dependencies: [],
    createdAt: 0, updatedAt: 0, attempt: 1, kind: 'implementation',
    assignee: 'impl-1', inScope: ['server/src/'], acceptance: ['Matches the approved requirements'],
    verify: ['pnpm vitest run server/src/routes/refund.test.ts'],
    changedPaths: ['server/src/routes/refund.ts'],
  }],
}

const englishReview = {
  id: 't2', subject: 'review-round-1', status: 'failed', dependencies: ['t1'],
  createdAt: 0, updatedAt: 0, attempt: 1, kind: 'review', round: 1,
  assignee: 'rev-1', verdict: 'needs_revision', reviewedTaskId: 't1',
  objective: 'Review whether the refund change satisfies the approved requirements',
  acceptance: ['The latest implementation meets the user goal', 'No unresolved blocker or high findings'],
  findings: [{
    id: 'F-9', severity: 'high', file: 'docs/api.md',
    problem: 'The documented refund payload omits the required idempotency key, so every example is rejected by the validator.',
    requiredFix: 'Add the key to the examples in docs/api.md and align the schema in server/src/validation/refund-schema.ts with the route in server/src/routes/refund.ts.',
  }],
}

const chineseReview = {
  ...englishReview,
  findings: [{
    id: 'F-zh', severity: 'high', file: 'docs/api.md',
    problem: '退款接口在没有幂等键时会重复扣款，示例文档也没提这个必填字段。',
    requiredFix: '补上幂等键校验；同时把示例文档改成实际能跑通的写法。',
  }],
}

const scopeCandidates = [
  'docs/api.md',
  'docs/migration-guide.md',
  'server/src/routes/refund.ts',
  'server/src/validation/refund-schema.ts',
  'README.md',
  'package.json',
  'server/src/',
]

const config = resolveJevConfig({ enabled: true, model: 'jev-latest', apiKeyEnv: 'JEV_KEY', timeoutMs: 20_000 })
const diagnostics = []

function report(label, hints, planned) {
  console.log(`\n=== ${label} ===`)
  console.log('  hints:', JSON.stringify(hints))
  if (planned === undefined) {
    console.log('  planned: (none)')
    return
  }
  for (const task of planned.created) {
    console.log(`  ${task.kind}: assignee=${task.assignee ?? '(none)'} inScope=${JSON.stringify(task.inScope ?? [])}`)
  }
  if (planned.escalated === true) console.log('  planned: ESCALATED')
}

// 1. English review, live decision call, no translator needed.
const english = createJevDecisions(config, { JEV_KEY: key }, {
  onDiagnostic: (message) => diagnostics.push(message),
})
const englishHints = await english.decide({ team, closed: englishReview, scopeCandidates })
report('live · English review', englishHints, planQualityFollowUp(team, englishReview, englishHints))

// 2. Chinese review with no translator: must abstain without calling the API.
let apiCalls = 0
const guarded = createJevDecisions(config, { JEV_KEY: key }, {
  fetch: (...args) => { apiCalls += 1; return globalThis.fetch(...args) },
  onDiagnostic: (message) => diagnostics.push(message),
})
const untranslated = await guarded.decide({ team, closed: chineseReview, scopeCandidates })
report('live · Chinese review, no translator route', untranslated, planQualityFollowUp(team, chineseReview, untranslated))
console.log(`  API calls attempted: ${apiCalls} (must be 0)`)

// 3. Chinese review through the boundary translator, then a live decision call.
const translatedSends = []
const translator = {
  available: true,
  async translate(slots) {
    // Stand-in for the real boundary translator: rewrite the CJK prose with
    // the English twin of the same finding and pass every already-English
    // slot through untouched.
    const english = {
      'review.objective': englishReview.objective,
      'review.acceptance[*]#0': englishReview.acceptance[0],
      'review.acceptance[*]#1': englishReview.acceptance[1],
      'findings[*].problem#0': englishReview.findings[0].problem,
      'findings[*].requiredFix#0': englishReview.findings[0].requiredFix,
    }
    const resolved = new Map()
    for (const slot of slots) {
      const keyed = slot.index === undefined ? slot.path : `${slot.path}#${slot.index}`
      resolved.set(keyed, english[keyed] ?? slot.value)
    }
    return resolved
  },
}
const translated = createJevDecisions(config, { JEV_KEY: key }, {
  translator,
  fetch: async (url, init) => {
    translatedSends.push(JSON.parse(init.body))
    return globalThis.fetch(url, init)
  },
  onDiagnostic: (message) => diagnostics.push(message),
})
const translatedHints = await translated.decide({ team, closed: chineseReview, scopeCandidates })
report('live · Chinese review, translated at the boundary', translatedHints, planQualityFollowUp(team, chineseReview, translatedHints))

const sentProse = JSON.stringify(translatedSends[0]?.state ?? {})
console.log(`\n  request state carries CJK: ${/[\u4E00-\u9FFF]/u.test(sentProse)} (must be false)`)
console.log(`  findings sent: ${JSON.stringify(translatedSends[0]?.state?.findings ?? [])}`)
console.log(`  questions sent: ${Object.keys(translatedSends[0]?.questions ?? {}).length}`)
console.log(`\n  diagnostics: ${diagnostics.length === 0 ? '(none)' : diagnostics.join(' | ')}`)