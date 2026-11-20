<script setup lang="ts">
// "Why is this post here?": the evidence behind one assignment, its path to the cluster's
// core entities, and a rationale that may only restate that evidence.
import { computed, ref } from 'vue'
import { api, type PostDetail } from '../lib/api'
import { WEAK_MARGIN, clusterById, clusterName, dashboard, isWeak, selectPost } from '../lib/dashboard'

const props = defineProps<{ detail: PostDetail }>()

const generating = ref(false)
const showText = ref(false)

const weak = computed(() => isWeak(props.detail))
const margin = computed(() => (props.detail.margin === null ? '–' : props.detail.margin.toFixed(2)))
const cluster = computed(() => clusterById(props.detail.cluster_id))

async function generate() {
  generating.value = true
  try {
    const { rationale, rationale_from_llm } = await api.generateRationale(dashboard.runId, props.detail.post)
    dashboard.detail = { ...props.detail, rationale, rationale_from_llm }
  } catch (error) {
    dashboard.error = `Rationale failed: ${(error as Error).message}`
  } finally {
    generating.value = false
  }
}
</script>

<template>
  <section class="card post" aria-labelledby="post-title">
    <header class="post__head">
      <div>
        <h2 id="post-title" class="title">
          <span class="mono">{{ detail.post }}</span> is in cluster {{ detail.cluster_id }}
        </h2>
        <p class="card__subtitle">
          {{ clusterName(cluster) }}
          <template v-if="detail.is_noise"> · HDBSCAN marked it noise; shown with its nearest cluster</template>
          <template v-if="detail.newsgroup"> · gold newsgroup: {{ detail.newsgroup }}</template>
        </p>
      </div>
    </header>

    <div class="strength" :class="weak ? 'strength--weak' : 'strength--strong'" role="status">
      <md-icon>{{ weak ? 'help' : 'verified' }}</md-icon>
      <div>
        <strong>{{ weak ? 'Weak evidence' : 'Strong evidence' }}</strong>
        <span>
          · {{ detail.term_hits.length }} cluster term{{ detail.term_hits.length === 1 ? '' : 's' }} in the text,
          margin {{ margin }}. Weak means a margin below {{ WEAK_MARGIN }} or at most one term: the
          assignments most likely to change when evidence is removed.
        </span>
      </div>
    </div>

    <div class="stats">
      <div class="stat">
        <span class="label">Cosine to own center</span>
        <span class="stat__value">{{ detail.own_centroid_sim.toFixed(2) }}</span>
      </div>
      <div class="stat">
        <span class="label">Margin vs cluster {{ detail.runner_up_cluster ?? '–' }}</span>
        <span class="stat__value">{{ margin }}</span>
      </div>
      <div class="stat" :title="detail.membership_prob === null ? 'Only HDBSCAN runs give membership probabilities.' : ''">
        <span class="label">Membership probability</span>
        <span class="stat__value">{{ detail.membership_prob === null ? 'n/a' : detail.membership_prob.toFixed(2) }}</span>
      </div>
    </div>

    <div class="group">
      <span class="label">Cluster terms in this post</span>
      <div class="tags">
        <span v-for="hit in detail.term_hits" :key="hit.term" class="tag tag--tonal">{{ hit.term }} × {{ hit.count }}</span>
        <span v-if="!detail.term_hits.length" class="muted">none</span>
      </div>
    </div>

    <div class="group">
      <span class="label">Nearest posts in the cluster</span>
      <md-chip-set>
        <md-assist-chip
          v-for="e in detail.exemplars"
          :key="e.post"
          :label="`${e.post} · ${e.similarity.toFixed(2)}`"
          @click="selectPost(e.post)"
        >
          <md-icon slot="icon">article</md-icon>
        </md-assist-chip>
      </md-chip-set>
    </div>

    <div v-if="detail.path" class="group">
      <span class="label">Path to the cluster's core ({{ detail.path.core.join(', ') }})</span>
      <ol v-if="detail.path.steps.length" class="path">
        <li v-if="detail.path.named_in_text" class="path__step">
          <span class="path__node path__node--post">this post</span>
          <span class="path__edge">names</span>
        </li>
        <li v-for="(s, i) in detail.path.steps" :key="i" class="path__step">
          <span v-if="s.relation" class="path__edge">
            {{ s.direction === 'backward' ? '←' : '' }}{{ s.relation }}{{ s.direction === 'forward' ? '→' : '' }}
          </span>
          <span class="path__node" :class="s.kind === 'post' ? 'path__node--post' : ''">{{ s.name }}</span>
        </li>
      </ol>
      <p v-else class="muted">
        {{ detail.path.names_any_entity
          ? "The entities this post names aren't connected to the core entities."
          : "This post names none of the cluster's graph entities." }}
      </p>
    </div>

    <div v-if="detail.entities.length" class="group">
      <span class="label">Entities extracted from this post</span>
      <div class="tags"><span v-for="e in detail.entities" :key="e" class="tag">{{ e }}</span></div>
    </div>

    <md-divider></md-divider>

    <div class="group">
      <div class="rationale-head">
        <span class="label">{{ detail.rationale_from_llm ? 'Rationale (LLM, checked against the evidence)' : 'Rationale (template from the evidence)' }}</span>
        <md-filled-tonal-button v-if="!detail.rationale_from_llm" :disabled="generating" @click="generate">
          <md-icon slot="icon">auto_awesome</md-icon>
          {{ generating ? 'Asking the LLM…' : 'Write with the LLM' }}
        </md-filled-tonal-button>
      </div>
      <p class="rationale">{{ detail.rationale }}</p>
    </div>

    <md-text-button class="read-toggle" @click="showText = !showText">
      <md-icon slot="icon">{{ showText ? 'expand_less' : 'expand_more' }}</md-icon>
      {{ showText ? 'Hide the post' : 'Read the post' }}
    </md-text-button>
    <pre v-if="showText" class="post__text">{{ detail.text }}</pre>
  </section>
</template>

<style scoped>
.title { font: var(--md-sys-typescale-title-large); }
.strength {
  display: flex;
  gap: var(--space-2);
  align-items: flex-start;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--md-sys-shape-corner-medium);
  font: var(--md-sys-typescale-body-medium);
}
.strength--strong { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.strength--weak { background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: var(--space-2); }
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--md-sys-shape-corner-medium);
  background: var(--md-sys-color-surface-container);
}
.stat__value { font: var(--md-sys-typescale-headline-small); font-variant-numeric: tabular-nums; }
.group { display: flex; flex-direction: column; gap: var(--space-2); }
.tags { display: flex; flex-wrap: wrap; gap: var(--space-2); }
.path { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-1); }
.path__step { display: inline-flex; align-items: center; gap: var(--space-1); }
.path__node {
  padding: 4px 12px;
  border-radius: var(--md-sys-shape-corner-full);
  background: var(--md-sys-color-primary-container);
  color: var(--md-sys-color-on-primary-container);
  font: var(--md-sys-typescale-label-large);
}
.path__node--post { background: var(--md-sys-color-surface-container-highest); color: var(--md-sys-color-on-surface); }
.path__edge { font: var(--md-sys-typescale-label-medium); color: var(--md-sys-color-on-surface-variant); }
.rationale-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); flex-wrap: wrap; }
.rationale { font: var(--md-sys-typescale-body-large); max-width: 75ch; }
.read-toggle { align-self: flex-start; }
.post__text {
  margin: 0;
  max-height: 360px;
  overflow: auto;
  padding: var(--space-3);
  border-radius: var(--md-sys-shape-corner-medium);
  background: var(--md-sys-color-surface-container);
  font: 13px/1.5 var(--font-mono);
  white-space: pre-wrap;
}
</style>
