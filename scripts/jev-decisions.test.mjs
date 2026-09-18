#!/usr/bin/env node
/**
 * Tests for the optional Jev decision layer and the boundary translation that
 * feeds it.
 *
 * The layer is advisory by construction, so the cases that matter most are
 * the DEGRADATION paths: no hints, a hint the roster cannot honor, a decision
 * service that fails, and non-English prose with no translation route must
 * all leave the pre-existing pure heuristics in charge. A regression here
 * would silently change how automatic repair rounds are planned for every
 * team, including teams that never enabled the layer.
 *
 * Run: node --test scripts/jev-decisions.test.mjs
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  classifyChangedPath,
  collectRepairScopeCandidates,
  planQualityFollowUp,
  repairScopeFromFindings,
} from '../lib/quality-gates.js'
import {
  acceptedChoice,
  acceptedNoul,
  buildRoutingQuestions,
  earlierFindings,
  applyTranslation,
  createJevDecisions,
  hintsFromAnswers,
  needsEnglishProjection,
  proseSlotKey,
  requiresTranslation,
  resolveJevConfig,
  topProbability,
} from '../lib/jev.js'
import { completeTranslation, parseTranslationObject } from '../lib/jev-translate.js'
import { Config } from '../lib/index.js'
import {
  describeCredentialOrigin,
  readLoginKeychain,
  resolveJevCredential,
} from '../lib/jev-credential.js'

function finding(extra = {}) {
  return { id: 'F1', severity: 'high', problem: 'a defect', requiredFix: 'fix it in codes.ts.', ...extra }
}

function member(name, role, extra = {}) {
  return { id: `member-${name}`, name, role, joinedAt: 0, status: 'idle', ...extra }
}

function team(extra = {}) {
  return {
    name: 't', id: 't', description: '', captainSessionId: 'captain',
    createdAt: 0, taskSeq: 3,
    members: [member('impl', 'engineer'), member('rev', 'reviewer'), member('docs', 'writer')],
    tasks: [{
      id: 't1', subject: 'impl', status: 'completed', dependencies: [],
      createdAt: 0, updatedAt: 0, attempt: 1, kind: 'implementation',
      assignee: 'impl', inScope: ['server/src/'], acceptance: ['done'], verify: ['pnpm test'],
    }],
    ...extra,
  }
}

function failedReview(extra = {}) {
  return {
    id: 't2', subject: 'review-round-1', status: 'failed', dependencies: ['t1'],
    createdAt: 0, updatedAt: 0, attempt: 1, kind: 'review', round: 1,
    assignee: 'rev', verdict: 'needs_revision', reviewedTaskId: 't1',
    objective: 'Review the change', acceptance: ['no blockers'],
    findings: [finding()],
    ...extra,
  }
}

function plannedRepair(result) {
  return result.created.find((task) => task.kind === 'repair')
}

function plannedReview(result) {
  return result.created.find((task) => task.kind === 'review')
}

// ── no-hint behaviour must be byte-identical to the pre-layer heuristic ──────

test('absent hints keep the positional assignee and the regex-derived scope', () => {
  const state = team()
  const planned = planQualityFollowUp(state, failedReview())
  const repair = plannedRepair(planned)
  assert.equal(repair.assignee, 'impl')
  assert.deepEqual(repair.inScope, ['codes.ts'])
  assert.equal(plannedReview(planned).assignee, 'rev')
})

test('empty hint fields are ignored rather than allowed to widen the contract', () => {
  const planned = planQualityFollowUp(
    team(),
    failedReview(),
    { repairInScope: [], implementer: '', reviewer: '' },
  )
  const repair = plannedRepair(planned)
  assert.deepEqual(repair.inScope, ['codes.ts'])
  assert.equal(repair.assignee, 'impl')
})

// ── hints replace a heuristic only when they can be honored ─────────────────

test('the default union policy keeps every path the prose names and adds the hint', () => {
  const repair = plannedRepair(planQualityFollowUp(
    team(),
    failedReview(),
    { repairInScope: ['server/src/routes/refund.ts'] },
  ))
  assert.deepEqual(repair.inScope, ['server/src/routes/refund.ts', 'codes.ts'])
})

test('a replace-policy hint takes over wholesale', () => {
  const repair = plannedRepair(planQualityFollowUp(
    team(),
    failedReview(),
    {
      scopePolicy: 'replace',
      repairInScope: ['server/src/routes/refund.ts', 'server/src/validation/refund-schema.ts', 'docs/api.md'],
    },
  ))
  assert.deepEqual(repair.inScope, [
    'server/src/routes/refund.ts',
    'server/src/validation/refund-schema.ts',
    'docs/api.md',
  ])
})

test('union adds a path the token scan cannot see, without the source fallback', () => {
  const chinese = failedReview({
    findings: [{
      id: 'F-zh', severity: 'high',
      problem: '退款接口在没有幂等键时会重复扣款。',
      requiredFix: '补上幂等键校验，并把示例文档改成能跑通的写法。',
    }],
  })
  const repair = plannedRepair(planQualityFollowUp(
    team(),
    chinese,
    { repairInScope: ['server/src/routes/refund.ts', 'docs/api.md'] },
  ))
  assert.deepEqual(repair.inScope, ['server/src/routes/refund.ts', 'docs/api.md'])
  const withoutHints = plannedRepair(planQualityFollowUp(team(), chinese))
  assert.deepEqual(withoutHints.inScope, ['server/src/'])
})

test('duplicate scope hints collapse so the contract stays a set', () => {
  const repair = plannedRepair(planQualityFollowUp(
    team(),
    failedReview(),
    { scopePolicy: 'replace', repairInScope: ['a.ts', 'a.ts', 'b.ts'] },
  ))
  assert.deepEqual(repair.inScope, ['a.ts', 'b.ts'])
})

test('a routing hint wins when it names a live member', () => {
  const planned = planQualityFollowUp(team(), failedReview(), { implementer: 'docs', reviewer: 'rev' })
  assert.equal(plannedRepair(planned).assignee, 'docs')
})

test('a routing hint naming a removed member falls back to the heuristic', () => {
  const state = team({ members: [member('impl', 'engineer'), member('rev', 'reviewer', { status: 'removed' }), member('docs', 'writer')] })
  const planned = planQualityFollowUp(state, failedReview(), { implementer: 'rev', reviewer: 'rev' })
  assert.equal(plannedRepair(planned).assignee, 'impl')
})

test('a routing hint may not assign the review to the implementer it judges', () => {
  const planned = planQualityFollowUp(team(), failedReview(), { implementer: 'impl', reviewer: 'impl' })
  assert.equal(plannedRepair(planned).assignee, 'impl')
  assert.notEqual(plannedReview(planned).assignee, 'impl')
})

test('a hint may not name the captain as a worker', () => {
  const planned = planQualityFollowUp(team(), failedReview(), { implementer: 'captain' })
  assert.equal(plannedRepair(planned).assignee, 'impl')
})

// ── semantic finding aliases keep the repair budget on the DEFECT ───────────

test('a renamed finding resets the repair budget without aliases', () => {
  const state = team({
    reviewPolicy: { maxRepairAttempts: 1, codeMaxRounds: 4 },
    tasks: [
      ...team().tasks,
      {
        id: 't3', subject: 'repair-round-1', status: 'completed', dependencies: ['t1'],
        createdAt: 0, updatedAt: 0, attempt: 1, kind: 'repair', round: 1,
        sourceTaskId: 't1', sourceFindingIds: ['F1'],
      },
    ],
  })
  const planned = planQualityFollowUp(state, failedReview({ findings: [finding({ id: 'F9' })] }))
  assert.ok(plannedRepair(planned) !== undefined, 'the budget reset and a new repair was planned')
})

test('an alias onto the earlier finding makes the same defect hit the budget', () => {
  const state = team({
    reviewPolicy: { maxRepairAttempts: 1, codeMaxRounds: 4 },
    tasks: [
      ...team().tasks,
      {
        id: 't3', subject: 'repair-round-1', status: 'completed', dependencies: ['t1'],
        createdAt: 0, updatedAt: 0, attempt: 1, kind: 'repair', round: 1,
        sourceTaskId: 't1', sourceFindingIds: ['F1'],
      },
    ],
  })
  const planned = planQualityFollowUp(
    state,
    failedReview({ findings: [finding({ id: 'F9' })] }),
    { findingAliases: { F9: ['F1'] } },
  )
  assert.equal(planned.escalated, true)
  assert.deepEqual(planned.created, [])
})

// ── candidate collection feeds the decision without becoming the decision ───

test('candidate collection unions observed files, declared scopes, and prose tokens', () => {
  const candidates = collectRepairScopeCandidates(team(), failedReview())
  assert.ok(candidates.includes('server/src'))
  assert.ok(candidates.includes('codes.ts'), `codes.ts missing from ${JSON.stringify(candidates)}`)
})

// ── the trailing-period dead-lock regression ────────────────────────────────

test('a sentence period is not glued onto a prose-derived path', () => {
  const scope = repairScopeFromFindings(
    [finding({ file: undefined, requiredFix: 'Fix the code in server/src/errors/codes.ts. See README.md for context.' })],
    ['src/'],
  )
  assert.ok(scope.includes('server/src/errors/codes.ts'), `path missing from ${JSON.stringify(scope)}`)
  assert.ok(!scope.includes('server/src/errors/codes.ts.'), `mangled path present in ${JSON.stringify(scope)}`)
  assert.equal(classifyChangedPath('server/src/errors/codes.ts', scope, []), 'in_scope')
})

// ── transport and abstention ────────────────────────────────────────────────

test('topProbability reads the distribution, and a noul answer reads yes/no', () => {
  assert.equal(topProbability({ type: 'choice', probabilities: { a: 0.2, b: 0.8 } }), 0.8)
  assert.equal(topProbability({ type: 'noul', noul: 0.84 }), 0.84)
  assert.equal(topProbability({ type: 'noul', noul: 0.1 }), 0.9)
  assert.equal(topProbability(undefined), 0)
})

test('an answer below the abstention floor is discarded', () => {
  const answer = { type: 'choice', choice: 'include', probabilities: { include: 0.53, exclude: 0.4, unknown: 0.07 } }
  assert.equal(acceptedChoice(answer, ['include', 'exclude', 'unknown'], 0.6), undefined)
  assert.equal(acceptedChoice(answer, ['include', 'exclude', 'unknown'], 0.5), 'include')
})

test('an explicit unknown option is always an abstention, whatever its probability', () => {
  const answer = { type: 'choice', choice: 'unknown', probabilities: { include: 0.1, unknown: 0.9 } }
  assert.equal(acceptedChoice(answer, ['include', 'unknown'], 0.6), undefined)
})

test('a choice outside the enumerated option set is rejected', () => {
  const answer = { type: 'choice', choice: 'invented', probabilities: { invented: 1 } }
  assert.equal(acceptedChoice(answer, ['include', 'unknown'], 0.6), undefined)
})

test('a coin-flip Noul is not agreement', () => {
  assert.equal(acceptedNoul({ type: 'noul', noul: 0.5 }, 0.6), undefined)
  assert.equal(acceptedNoul({ type: 'noul', noul: 0.5 }, 0.4), true)
})

test('hints fold only accepted answers', () => {
  const hints = hintsFromAnswers(
    {
      'scope::F1::a.ts': { type: 'choice', choice: 'include', probabilities: { include: 0.9, exclude: 0.05, unknown: 0.05 } },
      'scope::F1::b.ts': { type: 'choice', choice: 'include', probabilities: { include: 0.4, exclude: 0.35, unknown: 0.25 } },
      'owner::repair': { type: 'choice', choice: 'impl', probabilities: { impl: 0.95 } },
      'dup::F1::F0': { type: 'noul', noul: 0.89 },
    },
    {
      findingIds: ['F1'],
      scopeCandidates: ['a.ts', 'b.ts'],
      roster: ['impl'],
      routingTasks: ['repair'],
      earlierFindingIds: ['F0'],
      minProbability: 0.6,
      decisions: { repairScope: true, routing: true, dedup: true, scopePolicy: 'union' },
    },
  )
  assert.deepEqual(hints.repairInScope, ['a.ts'])
  assert.equal(hints.implementer, 'impl')
  assert.deepEqual(hints.findingAliases, { F1: ['F0'] })
})

test('a non-English payload with no translator abstains without calling the API', async () => {
  const diagnostics = []
  const decisions = createJevDecisions(
    resolveJevConfig({ enabled: true, model: 'jev-1.13.0' }),
    { JEV_API_KEY: 'test-key' },
    {
      fetch: () => { throw new Error('the API must not be called') },
      onDiagnostic: (message) => diagnostics.push(message),
    },
  )
  const hints = await decisions.decide({
    team: team(),
    closed: failedReview({ findings: [finding({ problem: '退款会重复扣款' })] }),
    scopeCandidates: ['a.ts'],
  })
  assert.deepEqual(hints, {})
  assert.ok(diagnostics.some((line) => line.includes('non-English prose')), diagnostics.join(' | '))
})

test('a failed decision call is a fail-open with a diagnostic', async () => {
  const diagnostics = []
  const decisions = createJevDecisions(
    resolveJevConfig({ enabled: true, model: 'jev-1.13.0' }),
    { JEV_API_KEY: 'test-key' },
    {
      fetch: () => Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({}) }),
      onDiagnostic: (message) => diagnostics.push(message),
    },
  )
  const hints = await decisions.decide({
    team: team(),
    closed: failedReview({ findings: [finding({ problem: 'english only' })] }),
    scopeCandidates: ['a.ts'],
  })
  assert.deepEqual(hints, {})
  assert.ok(diagnostics.some((line) => line.includes('503')), diagnostics.join(' | '))
})

test('a thrown transport error is a fail-open', async () => {
  const decisions = createJevDecisions(
    resolveJevConfig({ enabled: true, model: 'jev-1.13.0' }),
    { JEV_API_KEY: 'test-key' },
    { fetch: () => Promise.reject(new Error('offline')) },
  )
  const hints = await decisions.decide({
    team: team(),
    closed: failedReview({ findings: [finding({ problem: 'english only' })] }),
    scopeCandidates: ['a.ts'],
  })
  assert.deepEqual(hints, {})
})

test('an enabled layer with no resolvable credential is configured but abstains', async () => {
  const diagnostics = []
  const decisions = createJevDecisions(
    resolveJevConfig({ enabled: true, apiKeyEnv: 'MISSING_KEY' }),
    {},
    {
      credential: () => Promise.resolve(undefined),
      onDiagnostic: (message) => diagnostics.push(message),
    },
  )
  assert.equal(decisions.enabled, true)
  const hints = await decisions.decide({
    team: team(),
    closed: failedReview({ findings: [finding({ problem: 'english only' })] }),
    scopeCandidates: ['a.ts'],
  })
  assert.deepEqual(hints, {})
  assert.equal(diagnostics.filter((line) => line.includes('MISSING_KEY')).length, 1, 'a missing credential is reported once')
})

test('the default configuration is off', () => {
  const resolved = resolveJevConfig(undefined)
  assert.equal(resolved.enabled, false)
  assert.equal(resolved.minProbability, 0.6)
  assert.deepEqual(resolved.decisions, { repairScope: true, routing: true, dedup: true, scopePolicy: 'union' })
})

// ── boundary translation ────────────────────────────────────────────────────

test('English payloads are detected as needing no projection', () => {
  assert.equal(needsEnglishProjection('fix the parser'), false)
  assert.equal(needsEnglishProjection('修复解析器'), true)
  assert.equal(requiresTranslation([{ path: 'a', value: 'plain ascii' }]), false)
  assert.equal(requiresTranslation([{ path: 'a', value: '混合 prose' }]), true)
})

test('translated values are applied per slot and untranslated slots keep the original', () => {
  const slots = [
    { path: 'review.objective', value: '修复' },
    { path: 'findings[*].problem', index: 0, value: 'already english' },
  ]
  const resolved = applyTranslation(slots, new Map([[proseSlotKey(slots[0]), 'fix it']]))
  assert.equal(resolved.get('review.objective'), 'fix it')
  assert.equal(resolved.get('findings[*].problem#0'), 'already english')
})

test('a fenced JSON object parses and anything else does not', () => {
  assert.deepEqual(parseTranslationObject('```json\n{"a":"x"}\n```'), { a: 'x' })
  assert.deepEqual(parseTranslationObject('{"a":"x"}'), { a: 'x' })
  assert.equal(parseTranslationObject('here you go: {"a":"x"}'), undefined)
  assert.equal(parseTranslationObject('["a"]'), undefined)
  assert.equal(parseTranslationObject('{"a":""}'), undefined)
  assert.equal(parseTranslationObject('not json'), undefined)
})

test('a partial translation is rejected whole rather than mixed with the original', () => {
  const slots = [
    { path: 'findings[*].problem', index: 0, value: '一' },
    { path: 'findings[*].requiredFix', index: 0, value: '二' },
  ]
  assert.equal(completeTranslation(slots, { 'findings[*].problem#0': 'one' }), undefined)
  const complete = completeTranslation(slots, { 'findings[*].problem#0': 'one', 'findings[*].requiredFix#0': 'two' })
  assert.equal(complete.get('findings[*].requiredFix#0'), 'two')
})

// ── credential resolution never persists or prints the secret ───────────────

test('an explicit environment variable wins over the official name and the keychain', async () => {
  let keychainReads = 0
  const credential = await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: { JEV_API_KEY: 'primary', TYPESAFE_API_KEY: 'secondary' },
    platform: 'darwin',
    readKeychain: () => { keychainReads += 1; return Promise.resolve('keychain') },
  })
  assert.equal(credential.key, 'primary')
  assert.equal(keychainReads, 0)
})

test('the official SDK variable is used when the primary one is unset', async () => {
  const credential = await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: { TYPESAFE_API_KEY: 'secondary' },
    platform: 'darwin',
    readKeychain: () => Promise.resolve('keychain'),
  })
  assert.equal(credential.key, 'secondary')
  assert.deepEqual(credential.origin, { kind: 'environment', name: 'TYPESAFE_API_KEY' })
})

test('the keychain is the last resort, and only on macOS', async () => {
  const onMac = await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: {},
    platform: 'darwin',
    readKeychain: () => Promise.resolve('from-keychain'),
  })
  assert.deepEqual(onMac.origin, { kind: 'keychain', service: 'typesafe-jev', account: 'default' })
  const elsewhere = await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: {},
    platform: 'linux',
    readKeychain: () => Promise.resolve('from-keychain'),
  })
  assert.equal(elsewhere, undefined)
})

test('an empty or whitespace credential is not a credential', async () => {
  assert.equal(await resolveJevCredential({ apiKeyEnv: 'JEV_API_KEY', env: { JEV_API_KEY: '   ' }, platform: 'linux' }), undefined)
  assert.equal(await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: {},
    platform: 'darwin',
    readKeychain: () => Promise.resolve(''),
  }), undefined)
})

test('a keychain failure resolves to nothing rather than throwing', async () => {
  const credential = await resolveJevCredential({
    apiKeyEnv: 'JEV_API_KEY',
    env: {},
    platform: 'darwin',
    readKeychain: () => Promise.resolve(undefined),
  })
  assert.equal(credential, undefined)
})

test('the credential origin description never contains the secret', () => {
  const described = describeCredentialOrigin({ kind: 'keychain', service: 'typesafe-jev', account: 'default' })
  assert.equal(described, 'keychain item typesafe-jev/default')
  assert.equal(
    describeCredentialOrigin({ kind: 'environment', name: 'JEV_API_KEY' }),
    'environment variable JEV_API_KEY',
  )
})

test('the real keychain reader reports absence instead of throwing', async () => {
  const missing = await readLoginKeychain('dsh-agent-teams-nonexistent-service', 'none')
  assert.equal(missing, undefined)
})


// ── plugin config parsing ───────────────────────────────────────────────────
//
// Schemastery gives a nested `z.object()` an implicit `{}` default, so an
// optional nested block whose inner fields are required rejects a config that
// never mentioned it. That failure happens at plugin MOUNT time, which means a
// single missing `.const(undefined)` arm takes the whole harness row down.
// These cases are the ones an operator can actually write.

test('a minimal enable block parses and resolves', () => {
  const parsed = new Config({
    stateDir: '.agent-teams',
    memberProvider: 'spawn',
    jev: {
      enabled: true,
      model: 'jev-1.13.0',
      decisions: { repairScope: true, routing: true, dedup: true, scopePolicy: 'union' },
    },
  })
  const resolved = resolveJevConfig(parsed.jev)
  assert.equal(resolved.enabled, true)
  assert.equal(resolved.model, 'jev-1.13.0')
  assert.equal(resolved.decisions.scopePolicy, 'union')
  assert.equal(resolved.keychainService, 'typesafe-jev')
})

test('omitting the jev block entirely parses', () => {
  const parsed = new Config({ stateDir: '.agent-teams', memberProvider: 'spawn' })
  assert.equal(resolveJevConfig(parsed.jev).enabled, false)
})

test('an explicit translation route parses without requiring the rest', () => {
  const parsed = new Config({
    stateDir: '.agent-teams',
    memberProvider: 'spawn',
    jev: { enabled: true, translation: { provider: 'anthropic', model: 'claude-sonnet-4-5' } },
  })
  assert.equal(parsed.jev.translation.model, 'claude-sonnet-4-5')
  assert.equal(resolveJevConfig(parsed.jev).enabled, true)
})


// ── the dedup comparison set is the review's own lineage ────────────────────
//
// The first cut compared against findings on EVERY task. On a real team that
// pulled 22 unrelated findings out of two `work` tasks into the comparison:
// it diluted the question and, because the boundary translator renders every
// slot, it inflated the request until the layer abstained outright.

function lineageTeam(tasks) {
  return { ...team(), tasks }
}

test('unrelated work-task findings stay out of the comparison set', () => {
  const state = lineageTeam([
    { id: 't1', kind: 'implementation', status: 'completed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'impl' },
    { id: 't9', kind: 'work', status: 'completed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'docs',
      findings: [{ id: 'X1', severity: 'low', problem: 'unrelated', requiredFix: 'unrelated' }] },
    { id: 't11', kind: 'review', status: 'failed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'rev', round: 1, reviewedTaskId: 't1', verdict: 'needs_revision',
      findings: [{ id: 'N1', severity: 'high', problem: 'p', requiredFix: 'f' }] },
    { id: 't12', kind: 'repair', status: 'completed', dependencies: ['t1'], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'impl', sourceTaskId: 't1',
      findings: [{ id: 'N2', severity: 'low', problem: 'p2', requiredFix: 'f2' }] },
  ])
  const closed = state.tasks.find((t) => t.id === 't11')
  assert.deepEqual(earlierFindings(state, closed, ['N1']).map((f) => f.id), ['N2'])
})

test('a round-2 review still reaches the round-1 findings through the repair', () => {
  const state = lineageTeam([
    { id: 't1', kind: 'implementation', status: 'completed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'impl' },
    { id: 't11', kind: 'review', status: 'failed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'rev', round: 1, reviewedTaskId: 't1', verdict: 'needs_revision',
      findings: [{ id: 'N1', severity: 'high', problem: 'p', requiredFix: 'f' }] },
    { id: 't12', kind: 'repair', status: 'completed', dependencies: ['t1'], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'impl', sourceTaskId: 't1' },
    { id: 't13', kind: 'review', status: 'failed', dependencies: ['t12'], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'rev', round: 2, reviewedTaskId: 't12', verdict: 'needs_revision',
      findings: [{ id: 'R1', severity: 'medium', problem: 'r', requiredFix: 'rf' }] },
  ])
  const closed = state.tasks.find((t) => t.id === 't13')
  assert.deepEqual(earlierFindings(state, closed, ['R1']).map((f) => f.id), ['N1'])
})

test('incoming ids never echo back as earlier findings', () => {
  const state = lineageTeam([
    { id: 't11', kind: 'review', status: 'failed', dependencies: [], createdAt: 0, updatedAt: 0, attempt: 1, assignee: 'rev', round: 1, reviewedTaskId: 't1', verdict: 'needs_revision',
      findings: [{ id: 'N1', severity: 'high', problem: 'p', requiredFix: 'f' }] },
  ])
  const closed = state.tasks.find((t) => t.id === 't11')
  assert.deepEqual(earlierFindings(state, closed, ['N1']), [])
})

// ── routing questions state the intent, not a backwards constraint ──────────

test('routing questions carry per-task intent into the instructions', () => {
  const questions = buildRoutingQuestions(
    [
      { id: 'repair', kind: 'repair', objective: 'fix it', notes: 'PREFER_THE_IMPLEMENTER' },
      { id: 'review', kind: 'review', objective: 'review it', notes: 'KEEP_REVIEW_INDEPENDENT' },
    ],
    [{ name: 'impl', role: 'engineer' }, { name: 'rev', role: 'reviewer' }],
  )
  assert.match(questions['owner::repair'].instructions, /PREFER_THE_IMPLEMENTER/u)
  assert.match(questions['owner::review'].instructions, /KEEP_REVIEW_INDEPENDENT/u)
  assert.ok(Object.keys(questions['owner::repair'].criteria).includes('unknown'))
})


// ── a merged narrative still counts against the repair budget ───────────────

test('a finding that merges several earlier ones still exhausts their budget', () => {
  const state = team({
    reviewPolicy: { maxRepairAttempts: 1, codeMaxRounds: 6 },
    tasks: [
      ...team().tasks,
      {
        id: 't3', subject: 'repair-round-1', status: 'completed', dependencies: ['t1'],
        createdAt: 0, updatedAt: 0, attempt: 1, kind: 'repair', round: 1,
        sourceTaskId: 't1', sourceFindingIds: ['N3', 'N5'],
      },
    ],
  })
  const merged = failedReview({ findings: [finding({ id: 'T5' })] })
  // Set-equality alone cannot see this: the incoming key is T5, the recorded
  // key is N3,N5, so the budget never accumulates.
  assert.equal(planQualityFollowUp(state, merged).escalated, undefined)
  const budgeted = planQualityFollowUp(state, merged, { findingAliases: { T5: ['N3', 'N5'] } })
  assert.equal(budgeted.escalated, true)
  assert.deepEqual(budgeted.created, [])
})

test('a merged finding that also carries genuinely new work still opens a repair', () => {
  const state = team({
    reviewPolicy: { maxRepairAttempts: 1, codeMaxRounds: 6 },
    tasks: [
      ...team().tasks,
      {
        id: 't3', subject: 'repair-round-1', status: 'completed', dependencies: ['t1'],
        createdAt: 0, updatedAt: 0, attempt: 1, kind: 'repair', round: 1,
        sourceTaskId: 't1', sourceFindingIds: ['N3'],
      },
    ],
  })
  const planned = planQualityFollowUp(
    state,
    failedReview({ findings: [finding({ id: 'T5' }), finding({ id: 'T1' })] }),
    { findingAliases: { T5: ['N3'] } },
  )
  assert.equal(planned.escalated, undefined)
  assert.ok(planned.created.some((item) => item.kind === 'repair'))
})

test('every confirmed earlier finding is kept, not just the most confident', () => {
  const hints = hintsFromAnswers(
    {
      'dup::T5::N3': { type: 'noul', noul: 0.91 },
      'dup::T5::N5': { type: 'noul', noul: 0.88 },
      'dup::T5::N6': { type: 'noul', noul: 0.72 },
      'dup::T5::N7': { type: 'noul', noul: 0.55 },
    },
    {
      findingIds: ['T5'],
      scopeCandidates: [],
      roster: [],
      routingTasks: [],
      earlierFindingIds: ['N3', 'N5', 'N6', 'N7'],
      minProbability: 0.6,
      decisions: { repairScope: false, routing: false, dedup: true, scopePolicy: 'union' },
    },
  )
  assert.deepEqual(hints.findingAliases, { T5: ['N3', 'N5', 'N6'] })
})
