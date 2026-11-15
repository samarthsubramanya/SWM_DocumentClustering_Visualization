// Material 3 color: every --md-sys-color-* token is generated from one seed color, for a
// light and a dark scheme, so Material Web components and our own styles share one palette.
import {
  DynamicColor,
  Hct,
  MaterialDynamicColors,
  SchemeTonalSpot,
  argbFromHex,
  hexFromArgb,
} from '@material/material-color-utilities'
import { ref, watchEffect } from 'vue'

// ASU maroon: the project's school color, and a warm seed that reads well in both modes.
const SEED = '#8C1D40'

const ROLES = [
  'primary', 'onPrimary', 'primaryContainer', 'onPrimaryContainer', 'inversePrimary',
  'secondary', 'onSecondary', 'secondaryContainer', 'onSecondaryContainer',
  'tertiary', 'onTertiary', 'tertiaryContainer', 'onTertiaryContainer',
  'error', 'onError', 'errorContainer', 'onErrorContainer',
  'background', 'onBackground', 'surface', 'onSurface', 'surfaceVariant', 'onSurfaceVariant',
  'surfaceDim', 'surfaceBright', 'surfaceContainerLowest', 'surfaceContainerLow',
  'surfaceContainer', 'surfaceContainerHigh', 'surfaceContainerHighest',
  'inverseSurface', 'inverseOnSurface', 'outline', 'outlineVariant', 'shadow', 'scrim', 'surfaceTint',
] as const

const kebab = (role: string) => role.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())

function tokens(dark: boolean): Record<string, string> {
  const scheme = new SchemeTonalSpot(Hct.fromInt(argbFromHex(SEED)), dark, 0)
  const colors = MaterialDynamicColors as unknown as Record<string, DynamicColor>
  return Object.fromEntries(
    ROLES.map((role) => [`--md-sys-color-${kebab(role)}`, hexFromArgb(colors[role].getArgb(scheme))]),
  )
}

export type ThemeMode = 'system' | 'light' | 'dark'

const systemDark = window.matchMedia('(prefers-color-scheme: dark)')
const stored = (localStorage.getItem('theme') as ThemeMode | null) ?? 'system'

export const themeMode = ref<ThemeMode>(stored)
export const isDark = ref(false)

function apply() {
  isDark.value = themeMode.value === 'dark' || (themeMode.value === 'system' && systemDark.matches)
  const root = document.documentElement
  for (const [name, value] of Object.entries(tokens(isDark.value))) root.style.setProperty(name, value)
  root.style.colorScheme = isDark.value ? 'dark' : 'light'
}

export function startTheme() {
  systemDark.addEventListener('change', apply)
  watchEffect(() => {
    localStorage.setItem('theme', themeMode.value)
    apply()
  })
}

export function cycleTheme() {
  const order: ThemeMode[] = ['system', 'light', 'dark']
  themeMode.value = order[(order.indexOf(themeMode.value) + 1) % order.length]
}

// Categorical colors for clusters and entity types. They come from the same HCT color space
// as the scheme: evenly spread hues at a fixed chroma, with a tone that suits the mode.
const categorical = new Map<string, string>()

export function categoricalColor(index: number, dark: boolean): string {
  const key = `${index}:${dark}`
  if (!categorical.has(key)) {
    const hue = (index * 137.508 + 20) % 360 // golden angle: neighbours never share a hue
    categorical.set(key, hexFromArgb(Hct.from(hue, 48, dark ? 70 : 48).toInt()))
  }
  return categorical.get(key)!
}
