<script setup lang="ts">
import type { MdOutlinedSelect } from '@material/web/select/outlined-select.js'
import type { MdSwitch } from '@material/web/switch/switch.js'
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { dashboard, loadRuns } from './lib/dashboard'
import { cycleTheme, themeMode } from './lib/theme'

const destinations = [
  { to: '/explore', label: 'Explore', icon: 'bubble_chart' },
  { to: '/ask', label: 'Ask', icon: 'forum' },
  { to: '/evaluation', label: 'Evaluation', icon: 'fact_check' },
]

const route = useRoute()
const title = computed(() => (route.meta.title as string) ?? '')
const themeIcon = computed(() => ({ system: 'brightness_auto', light: 'light_mode', dark: 'dark_mode' })[themeMode.value])

// md-outlined-select ignores a value set before its options exist, which happens on first
// load; set it again once the run options have rendered.
const runSelect = ref<MdOutlinedSelect>()
watch(
  () => [dashboard.runId, dashboard.runs.length],
  async () => {
    await nextTick()
    if (runSelect.value) runSelect.value.value = dashboard.runId
  },
)

function onRunChange(event: Event) {
  dashboard.runId = (event.target as HTMLSelectElement).value
}

function onGoldChange(event: Event) {
  dashboard.showGold = (event.target as MdSwitch).selected
}

onMounted(loadRuns)
</script>

<template>
  <div class="shell">
    <!-- Navigation rail on medium and wider windows, a bottom bar on compact ones -->
    <nav class="rail" aria-label="Main">
      <div class="rail__brand" aria-hidden="true">
        <md-icon>hub</md-icon>
      </div>
      <RouterLink v-for="d in destinations" :key="d.to" :to="d.to" class="rail__item" active-class="is-active">
        <span class="rail__indicator">
          <md-ripple></md-ripple>
          <md-icon>{{ d.icon }}</md-icon>
        </span>
        <span class="rail__label">{{ d.label }}</span>
      </RouterLink>
    </nav>

    <div class="main">
      <header class="app-bar">
        <div class="app-bar__titles">
          <h1 class="app-bar__title">{{ title }}</h1>
          <p class="app-bar__subtitle">Explainable news clustering · CSE 573 · Group 17</p>
        </div>
        <div class="app-bar__actions">
          <md-outlined-select
            ref="runSelect"
            class="run-select"
            label="Clustering run"
            :value="dashboard.runId"
            @change="onRunChange"
          >
            <md-select-option v-for="run in dashboard.runs" :key="run.run_id" :value="run.run_id">
              <div slot="headline">{{ run.run_id }}</div>
            </md-select-option>
          </md-outlined-select>
          <label class="gold-toggle" title="Show the 20 Newsgroups labels. Off by default: the pipeline never sees them.">
            <md-switch :selected="dashboard.showGold" @change="onGoldChange"></md-switch>
            <span>Gold labels</span>
          </label>
          <md-icon-button :aria-label="`Theme: ${themeMode}`" :title="`Theme: ${themeMode}`" @click="cycleTheme">
            <md-icon>{{ themeIcon }}</md-icon>
          </md-icon-button>
        </div>
      </header>

      <md-linear-progress v-if="dashboard.loading" indeterminate class="app-progress"></md-linear-progress>
      <div v-if="dashboard.error" class="banner" role="alert">
        <md-icon>error</md-icon>
        <span>{{ dashboard.error }}</span>
        <md-text-button @click="dashboard.error = ''">Dismiss</md-text-button>
      </div>

      <main class="content">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  min-height: 100vh;
  background: var(--md-sys-color-surface);
}

.rail {
  position: sticky;
  top: 0;
  height: 100vh;
  width: 88px;
  flex: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) 0;
  background: var(--md-sys-color-surface-container);
}
.rail__brand {
  display: grid;
  place-items: center;
  width: 56px;
  height: 56px;
  margin-bottom: var(--space-2);
  border-radius: var(--md-sys-shape-corner-large);
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
}
.rail__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-decoration: none;
  color: var(--md-sys-color-on-surface-variant);
  font: var(--md-sys-typescale-label-medium);
}
.rail__indicator {
  position: relative;
  display: grid;
  place-items: center;
  width: 56px;
  height: 32px;
  border-radius: var(--md-sys-shape-corner-full);
  transition: background var(--md-sys-motion-duration-medium) var(--md-sys-motion-easing-emphasized);
}
.rail__item.is-active .rail__indicator {
  background: var(--md-sys-color-secondary-container);
  color: var(--md-sys-color-on-secondary-container);
}
.rail__item.is-active md-icon { font-variation-settings: 'FILL' 1; }
.rail__item.is-active { color: var(--md-sys-color-on-surface); }

.main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.app-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
}
.app-bar__title { font: var(--md-sys-typescale-headline-small); }
.app-bar__subtitle { font: var(--md-sys-typescale-body-small); color: var(--md-sys-color-on-surface-variant); }
.app-bar__actions { display: flex; align-items: center; gap: var(--space-3); flex-wrap: wrap; }
.run-select { min-width: 300px; }
.gold-toggle {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  font: var(--md-sys-typescale-label-large);
  color: var(--md-sys-color-on-surface-variant);
  cursor: pointer;
}

.app-progress { width: 100%; }
.banner {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0 var(--space-4) var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--md-sys-shape-corner-medium);
  background: var(--md-sys-color-error-container);
  color: var(--md-sys-color-on-error-container);
}
.banner span { flex: 1; }

.content {
  padding: 0 var(--space-4) var(--space-4);
  max-width: 1600px;
  width: 100%;
}

/* Compact windows: the rail becomes a bottom navigation bar */
@media (max-width: 599px) {
  .shell { flex-direction: column-reverse; }
  .rail {
    position: sticky;
    bottom: 0;
    top: auto;
    height: 80px;
    width: 100%;
    flex-direction: row;
    justify-content: space-around;
    padding: var(--space-2) 0;
    z-index: 2;
  }
  .rail__brand { display: none; }
  .app-bar, .content { padding-left: var(--space-3); padding-right: var(--space-3); }
  .run-select { min-width: 0; width: 100%; }
}
</style>
