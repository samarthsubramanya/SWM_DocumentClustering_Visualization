// Shared dashboard state: the chosen run, the selected cluster and post, and the gold toggle.
// Every view reads from here, so selecting a post anywhere (map, exemplar chip, typed id)
// updates the map, the cluster panel and the explanation together.
import { reactive, watch } from 'vue'
import { api, type Cluster, type Points, type PostDetail, type Run } from './api'

export const dashboard = reactive({
  runs: [] as Run[],
  runId: '',
  showGold: false,
  points: null as Points | null,
  clusters: [] as Cluster[],
  clusterId: null as number | null,
  post: '' as string, // opaque doc-… id
  detail: null as PostDetail | null,
  loading: false,
  loadingPost: false,
  error: '',
})

export function clusterById(id: number | null): Cluster | undefined {
  return dashboard.clusters.find((c) => c.cluster_id === id)
}

export function clusterName(cluster: Cluster | undefined): string {
  if (!cluster) return ''
  return cluster.label || cluster.keywords.slice(0, 3).map(([term]) => term).join(', ')
}

export async function loadRuns() {
  try {
    const { default: preferred, runs } = await api.runs()
    dashboard.runs = runs
    dashboard.runId = runs.some((r) => r.run_id === preferred) ? preferred : runs[0]?.run_id ?? ''
  } catch (error) {
    dashboard.error = `Could not reach the API: ${(error as Error).message}`
  }
}

async function loadRun() {
  if (!dashboard.runId) return
  dashboard.loading = true
  dashboard.error = ''
  try {
    const [points, clusters] = await Promise.all([
      api.points(dashboard.runId, dashboard.showGold),
      api.clusters(dashboard.runId, dashboard.showGold),
    ])
    dashboard.points = points
    dashboard.clusters = clusters
    if (!clusters.some((c) => c.cluster_id === dashboard.clusterId)) {
      dashboard.clusterId = clusters[0]?.cluster_id ?? null
    }
    if (dashboard.post) await selectPost(dashboard.post)
  } catch (error) {
    dashboard.error = (error as Error).message
  } finally {
    dashboard.loading = false
  }
}

// A new run or a flipped gold toggle changes what the server sends, so reload.
watch(() => [dashboard.runId, dashboard.showGold], loadRun)

export async function selectPost(post: string) {
  post = post.trim().toLowerCase()
  if (!post) return
  dashboard.post = post
  dashboard.loadingPost = true
  try {
    dashboard.detail = await api.post(dashboard.runId, post, dashboard.showGold)
    dashboard.clusterId = dashboard.detail.cluster_id
  } catch (error) {
    dashboard.detail = null
    dashboard.error = (error as Error).message
  } finally {
    dashboard.loadingPost = false
  }
}

export function selectCluster(id: number) {
  dashboard.clusterId = id
}

export function clearPost() {
  dashboard.post = ''
  dashboard.detail = null
}

// Evidence is "weak" with a small margin to the runner-up cluster or at most one cluster term:
// the assignments most likely to change when evidence is removed (same rule as ui/).
export const WEAK_MARGIN = 0.15
export function isWeak(detail: PostDetail): boolean {
  return (detail.margin !== null && detail.margin < WEAK_MARGIN) || detail.term_hits.length <= 1
}
