// Typed access to the FastAPI app: /api/v1/explore/* for the dashboard, /api/v1/query for chat.

export interface Run {
  run_id: string
  metrics: Record<string, number | null>
}

export interface Points {
  ids: string[]
  x: number[]
  y: number[]
  cluster: number[]
  noise: boolean[]
  newsgroup?: string[]
}

export interface Cluster {
  cluster_id: number
  size: number
  label: string | null
  keywords: [string, number][]
  newsgroups: { newsgroup: string; posts: number }[]
}

export interface GraphNode {
  id: string
  name: string
  type: string
  freq: number
  named_by_post: boolean
}

export interface GraphEdge {
  source: string
  target: string
  relation: string
}

export interface ClusterGraph {
  source: 'neo4j' | 'graphml' | 'none'
  hidden: number
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface PathStep {
  kind: 'post' | 'entity'
  name: string
  type?: string
  relation?: string
  direction?: 'forward' | 'backward'
}

export interface PostDetail {
  post: string
  cluster_id: number
  is_noise: boolean
  own_centroid_sim: number
  margin: number | null
  runner_up_cluster: number | null
  membership_prob: number | null
  term_hits: { term: string; count: number }[]
  exemplars: { post: string; similarity: number }[]
  entities: string[]
  path: { core: string[]; steps: PathStep[]; named_in_text: boolean; names_any_entity: boolean } | null
  rationale: string
  rationale_from_llm: boolean
  text: string
  newsgroup: string | null
}

export interface ReportEntry {
  name: string
  kind: 'md' | 'json'
  stamp: string
}

export interface AgentReply {
  answer: string
  agent: string | null
  metadata: Record<string, unknown>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    // FastAPI puts the reason in {detail}; fall back to the status line.
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

const explore = '/api/v1/explore'
const run = (runId: string) => `${explore}/runs/${encodeURIComponent(runId)}`

export const api = {
  runs: () => request<{ default: string; runs: Run[] }>(`${explore}/runs`),
  points: (runId: string, gold: boolean) => request<Points>(`${run(runId)}/points?gold=${gold}`),
  clusters: (runId: string, gold: boolean) => request<Cluster[]>(`${run(runId)}/clusters?gold=${gold}`),
  graph: (runId: string, clusterId: number, post?: string) =>
    request<ClusterGraph>(`${run(runId)}/clusters/${clusterId}/graph${post ? `?post=${post}` : ''}`),
  post: (runId: string, post: string, gold: boolean) =>
    request<PostDetail>(`${run(runId)}/posts/${post}?gold=${gold}`),
  generateRationale: (runId: string, post: string) =>
    request<{ rationale: string; rationale_from_llm: boolean }>(`${run(runId)}/posts/${post}/rationale`, { method: 'POST' }),
  reports: () => request<ReportEntry[]>(`${explore}/reports`),
  report: (name: string) => request<{ name: string; kind: string; content: unknown }>(`${explore}/reports/${encodeURIComponent(name)}`),
  health: () => request<{ status: string }>('/health'),
  ask: (query: string) =>
    request<AgentReply>('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }),
}
