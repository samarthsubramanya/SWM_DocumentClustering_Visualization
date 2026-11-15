<script setup lang="ts">
// Evaluation: the saved reports in eval/reports. The latest Text2Cypher run is summarized at
// the top (the mentor asked for validity before and after repair); any report can be read below.
import { computed, onMounted, ref, watch } from 'vue'
import { api, type ReportEntry } from '../lib/api'
import { renderMarkdown } from '../lib/markdown'

interface Text2CypherReport {
  n_questions: number
  n_informative: number
  validity: number
  validity_first_try: number
  execution_accuracy: number
  accuracy_by_pattern: Record<string, number>
  excluded: string[]
  results: { id: string; pattern: string; question: string; attempts: number; valid: boolean; valid_first_try: boolean; correct: boolean }[]
}

const reports = ref<ReportEntry[]>([])
const latestCypher = ref<{ name: string; report: Text2CypherReport } | null>(null)
const chosen = ref('')
const content = ref<{ kind: string; body: unknown } | null>(null)
const error = ref('')

const readable = computed(() => reports.value.filter((r) => r.kind === 'md'))

onMounted(async () => {
  try {
    reports.value = await api.reports()
    const cypher = reports.value.find((r) => r.kind === 'json' && r.name.startsWith('text2cypher-'))
    if (cypher) {
      const loaded = await api.report(cypher.name)
      latestCypher.value = { name: cypher.name, report: loaded.content as Text2CypherReport }
    }
    chosen.value = readable.value[0]?.name ?? ''
  } catch (e) {
    error.value = (e as Error).message
  }
})

watch(chosen, async (name) => {
  if (!name) return
  const loaded = await api.report(name)
  content.value = { kind: loaded.kind, body: loaded.content }
})

const percent = (value: number) => `${Math.round(value * 100)}%`

function status(row: Text2CypherReport['results'][number], excluded: string[]): string {
  if (excluded.includes(row.id)) return 'not scored'
  if (!row.valid) return 'invalid'
  return row.correct ? 'correct' : 'wrong'
}
</script>

<template>
  <div class="evaluation">
    <p v-if="error" class="muted">Could not load the reports: {{ error }}</p>

    <section v-if="latestCypher" class="card" aria-labelledby="cypher-title">
      <div>
        <h2 id="cypher-title" class="card__title">Text2Cypher: natural-language questions to graph queries</h2>
        <p class="card__subtitle">
          {{ latestCypher.name }} · {{ latestCypher.report.n_questions }} gold questions,
          {{ latestCypher.report.n_informative }} scored (gold answers that are empty or over 50 rows are left out)
        </p>
      </div>
      <div class="figures">
        <div class="figure">
          <span class="figure__value">{{ percent(latestCypher.report.validity_first_try) }}</span>
          <span class="label">valid on the first try</span>
        </div>
        <div class="figure">
          <span class="figure__value">{{ percent(latestCypher.report.validity) }}</span>
          <span class="label">valid after repair</span>
        </div>
        <div class="figure figure--primary">
          <span class="figure__value">{{ latestCypher.report.execution_accuracy.toFixed(3) }}</span>
          <span class="label">execution accuracy (same rows as the gold query)</span>
        </div>
      </div>
      <div class="table-wrap">
        <table class="data">
          <thead><tr><th>Id</th><th>Pattern</th><th>Question</th><th>Attempts</th><th>Result</th></tr></thead>
          <tbody>
            <tr v-for="row in latestCypher.report.results" :key="row.id">
              <td class="mono">{{ row.id }}</td>
              <td>{{ row.pattern }}</td>
              <td>{{ row.question }}</td>
              <td>{{ row.attempts }}</td>
              <td>
                <span class="result" :class="`result--${status(row, latestCypher.report.excluded).replace(' ', '-')}`">
                  {{ status(row, latestCypher.report.excluded) }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="reports">
      <nav class="card report-list" aria-label="Reports">
        <span class="label">Reports, newest first</span>
        <button
          v-for="r in readable"
          :key="r.name"
          class="report-link"
          :class="{ 'is-active': r.name === chosen }"
          @click="chosen = r.name"
        >
          <md-ripple></md-ripple>
          <md-icon>description</md-icon>
          <span>{{ r.name }}</span>
        </button>
      </nav>
      <article v-if="content" class="card report-body">
        <div class="prose" v-html="renderMarkdown(String(content.body))"></div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.evaluation { display: flex; flex-direction: column; gap: var(--space-3); }
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--space-2); }
.figure {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: var(--space-3);
  border-radius: var(--md-sys-shape-corner-large);
  background: var(--md-sys-color-surface-container);
}
.figure--primary { background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
.figure--primary .label { color: inherit; }
.figure__value { font: var(--md-sys-typescale-display-small); font-variant-numeric: tabular-nums; }
.result { padding: 2px 10px; border-radius: var(--md-sys-shape-corner-full); font: var(--md-sys-typescale-label-medium); }
.result--correct { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.result--wrong, .result--invalid { background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); }
.result--not-scored { color: var(--md-sys-color-on-surface-variant); border: 1px solid var(--md-sys-color-outline-variant); }

.reports { display: grid; grid-template-columns: minmax(240px, 320px) minmax(0, 1fr); gap: var(--space-3); align-items: start; }
@media (max-width: 839px) { .reports { grid-template-columns: minmax(0, 1fr); } }
.report-list { gap: var(--space-1); }
.report-link {
  position: relative;
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border: 0;
  border-radius: var(--md-sys-shape-corner-full);
  background: transparent;
  color: var(--md-sys-color-on-surface-variant);
  font: var(--md-sys-typescale-label-large);
  text-align: left;
  cursor: pointer;
  overflow-wrap: anywhere;
}
.report-link.is-active { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.report-body { max-width: 900px; }
</style>
