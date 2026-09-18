/**
 * Credential resolution for the Jev decision layer.
 *
 * The API key is never written to a file, a team's state directory, a command
 * line, or a log line. Resolution order is explicit environment override first
 * (CI and one-off runs), then the official SDK's own variable name, then the
 * macOS login keychain — which is where this machine already stores it, and
 * which is the only place a long-lived secret should live.
 *
 * Reading the keychain is why this module exists rather than a bare
 * `process.env[...]` read: the harness runs under launchd with a fixed
 * environment, so an env-only resolver would force every operator to either
 * restart the host with a secret in its environment or persist that secret in
 * a shell profile. Neither is acceptable.
 *
 * @module dsh-agent-teams/jev-credential
 */

import { execFile } from 'node:child_process'

/** Default keychain item holding the Jev API key. */
export const DEFAULT_KEYCHAIN_SERVICE = 'typesafe-jev'
/** Default keychain account for the Jev API key. */
export const DEFAULT_KEYCHAIN_ACCOUNT = 'default'
/** Official SDK variable name, honored as a secondary environment source. */
export const OFFICIAL_KEY_ENV = 'TYPESAFE_API_KEY'

/** Where a credential came from, for diagnostics that never include the value. */
export type JevCredentialOrigin =
  | { kind: 'environment', name: string }
  | { kind: 'keychain', service: string, account: string }

/** A resolved credential plus a non-sensitive description of its origin. */
export interface JevCredential {
  key: string
  origin: JevCredentialOrigin
}

/** Read one keychain item; injected so tests never touch the real keychain. */
export type KeychainReader = (
  service: string,
  account: string,
  signal?: AbortSignal,
) => Promise<string | undefined>

/**
 * Read the login keychain with `/usr/bin/security`.
 *
 * A missing item, a locked keychain, or a denied ACL all resolve to
 * `undefined` rather than throwing: the caller degrades to its heuristics, and
 * a credential problem must never look like a crash.
 */
export const readLoginKeychain: KeychainReader = (service, account, signal) =>
  new Promise((resolve) => {
    execFile(
      '/usr/bin/security',
      ['find-generic-password', '-s', service, '-a', account, '-w'],
      { signal, timeout: 5_000 },
      (error, stdout) => {
        if (error !== null) {
          resolve(undefined)
          return
        }
        const value = stdout.trim()
        resolve(value === '' ? undefined : value)
      },
    )
  })

/** Resolve the credential, or `undefined` when no source has one. */
export async function resolveJevCredential(input: {
  /** Primary environment variable name (from plugin config). */
  apiKeyEnv: string
  /** Environment to read; injected so callers can supply a scoped map. */
  env: Readonly<Record<string, string | undefined>>
  /** Platform gate for the keychain fallback; injected for tests. */
  platform?: string
  /** Keychain service. */
  service?: string
  /** Keychain account. */
  account?: string
  /** Keychain reader seam. */
  readKeychain?: KeychainReader
  /** Aborts an in-flight keychain read. */
  signal?: AbortSignal
}): Promise<JevCredential | undefined> {
  for (const name of [input.apiKeyEnv, OFFICIAL_KEY_ENV]) {
    const value = input.env[name]?.trim()
    if (value !== undefined && value !== '') return { key: value, origin: { kind: 'environment', name } }
  }
  const platform = input.platform ?? process.platform
  if (platform !== 'darwin') return undefined
  const service = input.service ?? DEFAULT_KEYCHAIN_SERVICE
  const account = input.account ?? DEFAULT_KEYCHAIN_ACCOUNT
  const read = input.readKeychain ?? readLoginKeychain
  const key = (await read(service, account, input.signal))?.trim()
  if (key === undefined || key === '') return undefined
  return { key, origin: { kind: 'keychain', service, account } }
}

/** Render a credential origin without ever rendering the credential. */
export function describeCredentialOrigin(origin: JevCredentialOrigin): string {
  return origin.kind === 'environment'
    ? `environment variable ${origin.name}`
    : `keychain item ${origin.service}/${origin.account}`
}