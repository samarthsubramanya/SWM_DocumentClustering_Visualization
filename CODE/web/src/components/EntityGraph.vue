<script setup lang="ts">
// A cluster's knowledge graph: its most mentioned entities and the relations between them.
// The layout is a force simulation run to rest once, then drawn as static SVG.
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationNodeDatum } from 'd3-force'
import { computed } from 'vue'
import type { ClusterGraph } from '../lib/api'
import { categoricalColor, isDark } from '../lib/theme'

const props = defineProps<{ graph: ClusterGraph }>()

const ENTITY_TYPES = ['Person', 'Organization', 'Location', 'Event', 'Artifact', 'Topic'] as const

type Node = SimulationNodeDatum & { id: string; name: string; type: string; freq: number; mine: boolean; r: number }

const WIDTH = 560
const HEIGHT = 420

function typeColor(type: string): string {
  const index = ENTITY_TYPES.indexOf(type as (typeof ENTITY_TYPES)[number])
  return categoricalColor(index < 0 ? 11 : index * 2 + 3, isDark.value)
}

const layout = computed(() => {
  const nodes: Node[] = props.graph.nodes.map((n) => ({
    id: n.id, name: n.name, type: n.type, freq: n.freq, mine: n.named_by_post,
    r: 6 + 2.2 * Math.min(n.freq, 10),
  }))
  const links = props.graph.edges.map((e) => ({ source: e.source, target: e.target, relation: e.relation }))

  forceSimulation(nodes)
    .force('link', forceLink<Node, (typeof links)[number] & { source: string | Node; target: string | Node }>(links as never)
      .id((d) => d.id).distance(70))
    .force('charge', forceManyBody().strength(-160))
    .force('center', forceCenter(WIDTH / 2, HEIGHT / 2))
    .force('collide', forceCollide<Node>().radius((d) => d.r + 14))
    // Most entities have no relation to the others; gentle gravity keeps them from drifting off.
    .force('x', forceX(WIDTH / 2).strength(0.07))
    .force('y', forceY(HEIGHT / 2).strength(0.07))
    .stop()
    .tick(300)

  // forceLink replaced the ids in links with node objects
  const edges = (links as unknown as { source: Node; target: Node; relation: string }[])

  // Fit whatever shape the layout settled into, labels included.
  const xs = nodes.flatMap((n) => [n.x! - n.r - 40, n.x! + n.r + 40])
  const ys = nodes.flatMap((n) => [n.y! - n.r - 8, n.y! + n.r + 22])
  const viewBox = nodes.length
    ? `${Math.min(...xs)} ${Math.min(...ys)} ${Math.max(...xs) - Math.min(...xs)} ${Math.max(...ys) - Math.min(...ys)}`
    : `0 0 ${WIDTH} ${HEIGHT}`
  return { nodes, edges, viewBox }
})

const typesShown = computed(() => ENTITY_TYPES.filter((t) => props.graph.nodes.some((n) => n.type === t)))
</script>

<template>
  <div class="graph">
    <svg :viewBox="layout.viewBox" role="img" aria-label="Entity graph of the selected cluster">
      <g class="edges">
        <line
          v-for="(e, i) in layout.edges"
          :key="i"
          :x1="e.source.x" :y1="e.source.y" :x2="e.target.x" :y2="e.target.y"
        >
          <title>{{ e.source.name }} —{{ e.relation }}→ {{ e.target.name }}</title>
        </line>
      </g>
      <g v-for="n in layout.nodes" :key="n.id" class="node" :class="{ 'node--mine': n.mine }">
        <circle :cx="n.x" :cy="n.y" :r="n.r" :fill="typeColor(n.type)">
          <title>{{ n.name }} · {{ n.type }} · {{ n.freq }} posts</title>
        </circle>
        <text :x="n.x" :y="n.y! + n.r + 14">{{ n.name }}</text>
      </g>
    </svg>
    <div class="legend">
      <span v-for="t in typesShown" :key="t" class="legend__item">
        <span class="legend__dot" :style="{ background: typeColor(t) }"></span>{{ t }}
      </span>
      <span v-if="graph.nodes.some((n) => n.named_by_post)" class="legend__item">
        <span class="legend__dot legend__dot--ring"></span>named by the selected post
      </span>
    </div>
  </div>
</template>

<style scoped>
.graph { display: flex; flex-direction: column; gap: var(--space-2); }
svg {
  width: 100%;
  height: 380px;
  border-radius: var(--md-sys-shape-corner-medium);
  background: var(--md-sys-color-surface-container-lowest);
}
.edges line { stroke: var(--md-sys-color-outline-variant); stroke-width: 1.4; }
.node circle { stroke: var(--md-sys-color-surface-container-lowest); stroke-width: 1.5; }
.node--mine circle { stroke: var(--md-sys-color-on-surface); stroke-width: 3.5; }
.node text {
  font: 500 11px var(--font);
  fill: var(--md-sys-color-on-surface);
  text-anchor: middle;
  paint-order: stroke;
  stroke: var(--md-sys-color-surface-container-lowest);
  stroke-width: 3px;
}
.node--mine text { font-weight: 700; }
.legend { display: flex; flex-wrap: wrap; gap: var(--space-1) var(--space-3); font: var(--md-sys-typescale-label-medium); color: var(--md-sys-color-on-surface-variant); }
.legend__item { display: inline-flex; align-items: center; gap: 6px; }
.legend__dot { width: 10px; height: 10px; border-radius: var(--md-sys-shape-corner-full); }
.legend__dot--ring { background: transparent; border: 2.5px solid var(--md-sys-color-on-surface); }
</style>
