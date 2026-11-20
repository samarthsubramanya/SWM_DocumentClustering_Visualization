<script setup lang="ts">
// Ask the agents: questions go to the harness router, which picks the cluster, explain or
// knowledge-graph agent. Knowledge-graph answers show the Cypher they ran and its rows.
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { api } from '../lib/api'
import { renderMarkdown } from '../lib/markdown'

interface Message {
  role: 'user' | 'assistant'
  text: string
  agent?: string | null
  metadata?: Record<string, unknown>
  failed?: boolean
}

interface CypherTrail {
  cypher: string
  attempts: number
  validation: { attempt: number; cypher: string; problems: string[] }[]
  rows?: Record<string, unknown>[]
}

const AGENTS: Record<string, string> = {
  cluster: 'Cluster agent',
  explain: 'Explain agent',
  kg: 'Knowledge-graph agent (Text2Cypher)',
  ingest: 'Ingest agent',
  none: 'No suitable agent',
}

const SUGGESTIONS = [
  'Why is year in braves?',
  'Describe the space cluster',
  'Which clusters have documents that mention nasa?',
  'Which people are affiliated with the fbi?',
  'Compare the clustering runs',
]

// Kept at module level so the conversation survives switching tabs.
const conversation = reactive<Message[]>([])

const draft = ref('')
const sending = ref(false)
const apiUp = ref<boolean | null>(null)
const log = ref<HTMLElement>()

const canSend = computed(() => draft.value.trim() !== '' && !sending.value && apiUp.value !== false)

onMounted(async () => {
  apiUp.value = await api.health().then(() => true, () => false)
})

function cypherTrail(message: Message): CypherTrail | null {
  const meta = message.metadata as Partial<CypherTrail> | undefined
  return meta?.cypher ? (meta as CypherTrail) : null
}

function columns(rows: Record<string, unknown>[]): string[] {
  return rows.length ? Object.keys(rows[0]) : []
}

async function send(question = draft.value) {
  question = question.trim()
  if (!question || sending.value) return
  conversation.push({ role: 'user', text: question })
  draft.value = ''
  sending.value = true
  await scrollDown()
  try {
    const reply = await api.ask(question)
    conversation.push({ role: 'assistant', text: reply.answer, agent: reply.agent, metadata: reply.metadata })
  } catch (error) {
    conversation.push({ role: 'assistant', text: `The request failed: ${(error as Error).message}`, failed: true })
  } finally {
    sending.value = false
    await scrollDown()
  }
}

async function scrollDown() {
  await nextTick()
  log.value?.lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'end' })
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="ask">
    <div v-if="apiUp === false" class="card card--outlined notice">
      <md-icon>cloud_off</md-icon>
      <span>The agent API is not reachable. Start it with <code class="mono">uvicorn src.api.app:app --port 8000</code>.</span>
    </div>

    <section v-if="!conversation.length" class="intro">
      <h2 class="intro__title">Ask about clusters, posts or the knowledge graph</h2>
      <p class="muted">A router reads your question and hands it to the agent that can answer it.</p>
      <md-chip-set>
        <md-suggestion-chip v-for="s in SUGGESTIONS" :key="s" :label="s" @click="send(s)"></md-suggestion-chip>
      </md-chip-set>
    </section>

    <ol ref="log" class="log" aria-live="polite">
      <li v-for="(m, i) in conversation" :key="i" class="message" :class="`message--${m.role}`">
        <div v-if="m.role === 'user'" class="bubble bubble--user">{{ m.text }}</div>

        <div v-else class="bubble bubble--assistant" :class="{ 'bubble--failed': m.failed }">
          <div class="prose" v-html="renderMarkdown(m.text)"></div>

          <template v-if="cypherTrail(m)">
            <div class="cypher">
              <span class="label">
                Generated Cypher · {{ cypherTrail(m)!.rows ? 'valid' : 'rejected' }} after
                {{ cypherTrail(m)!.attempts }} attempt{{ cypherTrail(m)!.attempts === 1 ? '' : 's' }} · read-only transaction
              </span>
              <pre class="code">{{ cypherTrail(m)!.cypher }}</pre>
              <details v-if="cypherTrail(m)!.validation.length" class="repairs">
                <summary>Rejected by validation ({{ cypherTrail(m)!.validation.length }})</summary>
                <div v-for="r in cypherTrail(m)!.validation" :key="r.attempt" class="repair">
                  <pre class="code">{{ r.cypher }}</pre>
                  <ul><li v-for="p in r.problems" :key="p">{{ p }}</li></ul>
                </div>
              </details>
              <div v-if="cypherTrail(m)!.rows?.length" class="table-wrap rows">
                <table class="data">
                  <thead><tr><th v-for="c in columns(cypherTrail(m)!.rows!)" :key="c">{{ c }}</th></tr></thead>
                  <tbody>
                    <tr v-for="(row, r) in cypherTrail(m)!.rows!.slice(0, 50)" :key="r">
                      <td v-for="c in columns(cypherTrail(m)!.rows!)" :key="c">{{ row[c] }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </template>

          <span v-if="m.agent" class="handled-by">
            <md-icon>smart_toy</md-icon>{{ AGENTS[m.agent] ?? m.agent }}
          </span>
        </div>
      </li>
      <li v-if="sending" class="message message--assistant">
        <div class="bubble bubble--assistant thinking">
          <md-circular-progress indeterminate></md-circular-progress>
          <span class="muted">Routing your question to an agent…</span>
        </div>
      </li>
    </ol>

    <form class="composer" @submit.prevent="send()">
      <md-outlined-text-field
        class="composer__field"
        label="Your question"
        placeholder="e.g. why is doc-3f9a1c2e in its cluster?"
        :value="draft"
        :disabled="apiUp === false"
        @input="draft = ($event.target as HTMLInputElement).value"
        @keydown="onKeydown"
      ></md-outlined-text-field>
      <md-filled-icon-button type="submit" :disabled="!canSend" aria-label="Send">
        <md-icon>send</md-icon>
      </md-filled-icon-button>
    </form>
  </div>
</template>

<style scoped>
.ask { display: flex; flex-direction: column; gap: var(--space-3); max-width: 960px; margin: 0 auto; min-height: calc(100vh - 140px); }
.notice { flex-direction: row; align-items: center; }
.intro { display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-4) 0; }
.intro__title { font: var(--md-sys-typescale-headline-small); }
.log { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--space-3); flex: 1; }
.message { display: flex; }
.message--user { justify-content: flex-end; }
.bubble { max-width: min(760px, 100%); padding: var(--space-2) var(--space-3); display: flex; flex-direction: column; gap: var(--space-2); min-width: 0; }
.bubble--user {
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
  border-radius: var(--md-sys-shape-corner-large) var(--md-sys-shape-corner-large) var(--md-sys-shape-corner-extra-small) var(--md-sys-shape-corner-large);
  font: var(--md-sys-typescale-body-large);
}
.bubble--assistant {
  background: var(--md-sys-color-surface-container-low);
  border-radius: var(--md-sys-shape-corner-large) var(--md-sys-shape-corner-large) var(--md-sys-shape-corner-large) var(--md-sys-shape-corner-extra-small);
  width: 100%;
}
.bubble--failed { background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); }
.thinking { flex-direction: row; align-items: center; --md-circular-progress-size: 24px; }
.cypher { display: flex; flex-direction: column; gap: var(--space-2); }
.code {
  margin: 0;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--md-sys-shape-corner-small);
  background: var(--md-sys-color-surface-container-highest);
  font: 13px/1.5 var(--font-mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.repairs summary { cursor: pointer; font: var(--md-sys-typescale-label-large); color: var(--md-sys-color-on-surface-variant); }
.repair { display: flex; flex-direction: column; gap: var(--space-1); margin-top: var(--space-2); }
.repair ul { margin: 0; padding-left: var(--space-4); color: var(--md-sys-color-error); }
.rows { max-height: 280px; overflow-y: auto; }
.handled-by { display: inline-flex; align-items: center; gap: 6px; font: var(--md-sys-typescale-label-medium); color: var(--md-sys-color-on-surface-variant); }
.handled-by md-icon { font-size: 18px; }
.composer {
  position: sticky;
  bottom: var(--space-3);
  display: flex;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-2);
  border-radius: var(--md-sys-shape-corner-extra-large);
  background: var(--md-sys-color-surface-container-high);
}
.composer__field { flex: 1; }
@media (max-width: 599px) { .composer { bottom: 96px; } }
</style>
