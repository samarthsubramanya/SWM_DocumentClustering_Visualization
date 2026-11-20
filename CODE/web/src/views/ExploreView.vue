<script setup lang="ts">
// Explore: run scores, the map of posts, the selected cluster and the selected post.
import { computed, ref, watch } from 'vue'
import ClusterMap from '../components/ClusterMap.vue'
import ClusterPanel from '../components/ClusterPanel.vue'
import PostCard from '../components/PostCard.vue'
import { clearPost, clusterById, clusterName, dashboard, selectPost } from '../lib/dashboard'

const METRICS: [key: string, label: string, hint: string][] = [
  ['nmi', 'NMI', 'Agreement with the 20 newsgroups (0–1)'],
  ['ari', 'ARI', 'Pairwise agreement, corrected for chance'],
  ['purity', 'Purity', 'Share of posts in their cluster’s majority newsgroup'],
  ['silhouette', 'Silhouette', 'Cluster separation; favors k-means / agglomerative'],
  ['dbcv', 'DBCV', 'Density-based validity without labels; favors HDBSCAN'],
  ['noise_rate', 'Noise', 'Share of posts HDBSCAN left unclustered'],
]

const run = computed(() => dashboard.runs.find((r) => r.run_id === dashboard.runId))
const cluster = computed(() => clusterById(dashboard.clusterId))

const typed = ref(dashboard.post)
watch(() => dashboard.post, (post) => (typed.value = post))

function format(value: number | null | undefined): string {
  return value === null || value === undefined ? '–' : value.toFixed(3)
}

function openTyped() {
  if (typed.value.trim()) selectPost(typed.value)
}
</script>

<template>
  <div class="explore">
    <section class="metrics" aria-label="Scores of this run">
      <div v-for="[key, label, hint] in METRICS" :key="key" class="metric" :title="hint">
        <span class="label">{{ label }}</span>
        <span class="metric__value">{{ format(run?.metrics[key]) }}</span>
      </div>
    </section>
    <p class="card__subtitle">Newsgroup labels are used only to compute these scores, never by the pipeline.</p>

    <div class="layout">
      <section class="card map-card" aria-labelledby="map-title">
        <div class="map-card__head">
          <div>
            <h2 id="map-title" class="card__title">Map of posts</h2>
            <p class="card__subtitle">Similar posts sit close together. Click a dot to see why it is in its cluster.</p>
          </div>
          <form class="post-search" @submit.prevent="openTyped">
            <md-outlined-text-field
              label="Post id"
              placeholder="doc-3f9a1c2e"
              :value="typed"
              @input="typed = ($event.target as HTMLInputElement).value"
              @keydown.enter.prevent="openTyped"
            >
              <md-icon slot="leading-icon">search</md-icon>
            </md-outlined-text-field>
            <md-filled-tonal-button type="submit">Open</md-filled-tonal-button>
          </form>
        </div>
        <ClusterMap
          v-if="dashboard.points"
          :points="dashboard.points"
          :cluster-id="dashboard.clusterId"
          :selected="dashboard.post"
          :cluster-label="cluster ? `${cluster.cluster_id} — ${clusterName(cluster)}` : ''"
          @pick="selectPost"
        />
        <div v-else class="placeholder">Loading the map…</div>
      </section>

      <ClusterPanel v-if="cluster" :cluster="cluster" class="cluster-pane" />
    </div>

    <md-linear-progress v-if="dashboard.loadingPost" indeterminate></md-linear-progress>
    <PostCard v-if="dashboard.detail" :detail="dashboard.detail" />
    <section v-else class="card card--outlined empty">
      <md-icon>touch_app</md-icon>
      <p>Pick a post on the map, or type its id, to see the evidence behind its cluster.</p>
    </section>
    <md-text-button v-if="dashboard.detail" class="clear" @click="clearPost">Clear selection</md-text-button>
  </div>
</template>

<style scoped>
.explore { display: flex; flex-direction: column; gap: var(--space-3); }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--space-2); }
.metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--md-sys-shape-corner-large);
  background: var(--md-sys-color-surface-container);
}
.metric__value { font: var(--md-sys-typescale-headline-small); font-variant-numeric: tabular-nums; }

/* List–detail: the map leads, the cluster pane supports it; stacked on narrower windows. */
.layout { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: var(--space-3); align-items: start; }
@media (max-width: 1199px) { .layout { grid-template-columns: minmax(0, 1fr); } }

.map-card__head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--space-3); align-items: flex-end; }
.post-search { display: flex; gap: var(--space-2); align-items: center; }
.post-search md-outlined-text-field { width: 220px; }
.placeholder { height: 520px; display: grid; place-items: center; color: var(--md-sys-color-on-surface-variant); }
.empty { flex-direction: row; align-items: center; color: var(--md-sys-color-on-surface-variant); }
.clear { align-self: flex-start; }
</style>
