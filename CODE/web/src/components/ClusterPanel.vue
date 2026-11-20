<script setup lang="ts">
// The selected cluster: its label, the terms that define it and its knowledge graph.
import { computed, ref, watch } from 'vue'
import { api, type Cluster, type ClusterGraph } from '../lib/api'
import { clusterName, dashboard, selectCluster } from '../lib/dashboard'
import EntityGraph from './EntityGraph.vue'

const props = defineProps<{ cluster: Cluster }>()

const graph = ref<ClusterGraph | null>(null)
const loadingGraph = ref(false)

// Ring the selected post's entities only when that post belongs to this cluster.
const postHere = computed(() =>
  dashboard.detail?.cluster_id === props.cluster.cluster_id ? dashboard.post : undefined,
)

watch(
  () => [dashboard.runId, props.cluster.cluster_id, postHere.value] as const,
  async ([runId, clusterId, post]) => {
    loadingGraph.value = true
    try {
      graph.value = await api.graph(runId, clusterId, post)
    } catch {
      graph.value = null
    } finally {
      loadingGraph.value = false
    }
  },
  { immediate: true },
)

const topWeight = computed(() => props.cluster.keywords[0]?.[1] || 1)

function onClusterChange(event: Event) {
  selectCluster(Number((event.target as HTMLSelectElement).value))
}
</script>

<template>
  <section class="card" aria-labelledby="cluster-title">
    <md-outlined-select label="Cluster" :value="String(cluster.cluster_id)" @change="onClusterChange">
      <md-select-option v-for="c in dashboard.clusters" :key="c.cluster_id" :value="String(c.cluster_id)">
        <div slot="headline">{{ c.cluster_id }} — {{ clusterName(c) }}</div>
        <div slot="supporting-text">{{ c.size }} posts</div>
      </md-select-option>
    </md-outlined-select>

    <div>
      <h2 id="cluster-title" class="title">{{ cluster.label || 'No knowledge-graph label yet' }}</h2>
      <p class="card__subtitle">
        {{ cluster.size }} posts · label = top PageRank entities of the cluster graph, top term in parentheses
      </p>
    </div>

    <div v-if="dashboard.showGold && cluster.newsgroups.length" class="newsgroups">
      <span class="label">Newsgroups here (reference only)</span>
      <span v-for="g in cluster.newsgroups" :key="g.newsgroup" class="tag">
        {{ g.newsgroup }} · {{ Math.round((100 * g.posts) / cluster.size) }}%
      </span>
    </div>

    <div class="terms">
      <span class="label">Top terms (c-TF-IDF)</span>
      <div v-for="[term, weight] in cluster.keywords.slice(0, 10)" :key="term" class="term">
        <span class="term__name">{{ term }}</span>
        <span class="term__bar" :style="{ width: `${(100 * weight) / topWeight}%` }"></span>
        <span class="term__weight">{{ weight.toFixed(3) }}</span>
      </div>
    </div>

    <md-divider></md-divider>

    <div class="graph-head">
      <span class="label">Knowledge graph</span>
      <md-circular-progress v-if="loadingGraph" indeterminate class="spinner"></md-circular-progress>
    </div>
    <EntityGraph v-if="graph && graph.nodes.length" :graph="graph" />
    <p v-else-if="!loadingGraph" class="muted">No knowledge graph for this cluster yet (scripts/05_kg.py).</p>
    <p v-if="graph && graph.nodes.length" class="card__subtitle">
      Top {{ graph.nodes.length }} entities by number of posts ({{ graph.hidden }} more hidden: addresses,
      unconnected or rare). Source: {{ graph.source }}.
    </p>
  </section>
</template>

<style scoped>
.title { font: var(--md-sys-typescale-title-large); text-wrap: balance; }
.newsgroups { display: flex; flex-wrap: wrap; gap: var(--space-2); align-items: center; }
.newsgroups .label { width: 100%; }
.terms { display: flex; flex-direction: column; gap: 6px; }
.term {
  display: grid;
  grid-template-columns: 96px 1fr 44px;
  align-items: center;
  gap: var(--space-2);
  font: var(--md-sys-typescale-body-medium);
}
.term__name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.term__bar {
  height: 10px;
  border-radius: var(--md-sys-shape-corner-full);
  background: var(--md-sys-color-primary);
  transition: width var(--md-sys-motion-duration-medium) var(--md-sys-motion-easing-emphasized);
}
.term__weight { font: var(--md-sys-typescale-label-small); color: var(--md-sys-color-on-surface-variant); text-align: right; font-variant-numeric: tabular-nums; }
.graph-head { display: flex; align-items: center; justify-content: space-between; }
.spinner { --md-circular-progress-size: 24px; }
</style>
