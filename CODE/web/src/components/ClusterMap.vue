<script setup lang="ts">
// The embedding map: every post as a dot at its UMAP position. Drawn on a canvas because a
// run has 10–17k posts. Ctrl/⌘ + scroll (or pinch) zooms, drag pans, a click opens a post.
// Plain scrolling is left to the page so the map never traps it.
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Points } from '../lib/api'
import { categoricalColor, isDark } from '../lib/theme'

const props = defineProps<{
  points: Points
  clusterId: number | null
  selected: string
  clusterLabel: string
}>()
const emit = defineEmits<{ pick: [post: string] }>()

const host = ref<HTMLDivElement>()
const canvas = ref<HTMLCanvasElement>()
const hover = ref<{ index: number; left: number; top: number } | null>(null)

// View transform on top of the "fit everything" scale: zoom factor and pan offset in pixels.
const view = { zoom: 1, panX: 0, panY: 0 }
let size = { width: 0, height: 0 }
let drag: { x: number; y: number; moved: boolean } | null = null

const bounds = computed(() => {
  const { x, y } = props.points
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (let i = 0; i < x.length; i++) {
    if (x[i] < minX) minX = x[i]
    if (x[i] > maxX) maxX = x[i]
    if (y[i] < minY) minY = y[i]
    if (y[i] > maxY) maxY = y[i]
  }
  return { minX, maxX, minY, maxY }
})

const selectedIndex = computed(() => props.points.ids.indexOf(props.selected))

function toScreen(px: number, py: number): [number, number] {
  const { minX, maxX, minY, maxY } = bounds.value
  const pad = 24
  const scale = Math.min((size.width - 2 * pad) / (maxX - minX || 1), (size.height - 2 * pad) / (maxY - minY || 1))
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  // y grows upwards in UMAP space and downwards on screen
  return [
    size.width / 2 + (px - cx) * scale * view.zoom + view.panX,
    size.height / 2 - (py - cy) * scale * view.zoom + view.panY,
  ]
}

function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

function draw() {
  const ctx = canvas.value?.getContext('2d')
  if (!ctx || !size.width) return
  const ratio = window.devicePixelRatio || 1
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
  ctx.clearRect(0, 0, size.width, size.height)

  const { x, y, cluster } = props.points
  const muted = token('--md-sys-color-outline')
  const highlight = props.clusterId
  const radius = Math.min(3.2, 1.6 * Math.sqrt(view.zoom))

  // Other clusters first, faded, so the selected one is drawn on top in full color.
  ctx.globalAlpha = highlight === null ? 0.85 : 0.28
  for (let i = 0; i < x.length; i++) {
    if (cluster[i] === highlight) continue
    const [sx, sy] = toScreen(x[i], y[i])
    ctx.fillStyle = highlight === null ? categoricalColor(cluster[i] + 1, isDark.value) : muted
    ctx.fillRect(sx - radius / 2, sy - radius / 2, radius, radius)
  }

  let top = Infinity
  let sumX = 0
  let count = 0
  if (highlight !== null) {
    ctx.globalAlpha = 0.9
    ctx.fillStyle = categoricalColor(highlight + 1, isDark.value)
    for (let i = 0; i < x.length; i++) {
      if (cluster[i] !== highlight) continue
      const [sx, sy] = toScreen(x[i], y[i])
      ctx.beginPath()
      ctx.arc(sx, sy, radius * 0.8, 0, Math.PI * 2)
      ctx.fill()
      sumX += sx
      count++
      top = Math.min(top, sy)
    }
  }
  ctx.globalAlpha = 1

  if (count && props.clusterLabel) drawLabel(ctx, props.clusterLabel, sumX / count, top - 10)
  if (selectedIndex.value >= 0) {
    const i = selectedIndex.value
    drawStar(ctx, ...toScreen(x[i], y[i]))
  }
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, cx: number, bottom: number) {
  ctx.font = '500 12px "Roboto Flex", sans-serif'
  const width = ctx.measureText(text).width + 20
  const left = Math.max(4, Math.min(size.width - width - 4, cx - width / 2))
  const topEdge = Math.max(4, bottom - 24)
  ctx.fillStyle = token('--md-sys-color-inverse-surface')
  ctx.beginPath()
  ctx.roundRect(left, topEdge, width, 24, 8)
  ctx.fill()
  ctx.fillStyle = token('--md-sys-color-inverse-on-surface')
  ctx.textBaseline = 'middle'
  ctx.fillText(text, left + 10, topEdge + 12)
}

function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  const outer = 11
  const inner = 4.6
  ctx.beginPath()
  for (let k = 0; k < 10; k++) {
    const r = k % 2 ? inner : outer
    const angle = -Math.PI / 2 + (k * Math.PI) / 5
    ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle))
  }
  ctx.closePath()
  // highest contrast on any palette: ink-colored star with a halo in the background color
  ctx.fillStyle = token('--md-sys-color-on-surface')
  ctx.strokeStyle = token('--md-sys-color-surface')
  ctx.lineWidth = 2
  ctx.fill()
  ctx.stroke()
}

function nearest(mx: number, my: number, within = 8): number {
  const { x, y } = props.points
  let best = -1
  let bestDistance = within * within
  for (let i = 0; i < x.length; i++) {
    const [sx, sy] = toScreen(x[i], y[i])
    const d = (sx - mx) ** 2 + (sy - my) ** 2
    if (d < bestDistance) {
      bestDistance = d
      best = i
    }
  }
  return best
}

function local(event: MouseEvent): [number, number] {
  const rect = canvas.value!.getBoundingClientRect()
  return [event.clientX - rect.left, event.clientY - rect.top]
}

function zoomAt(mx: number, my: number, factor: number) {
  const zoom = Math.min(40, Math.max(0.8, view.zoom * factor))
  // keep the point under the cursor in place while zooming
  const applied = zoom / view.zoom
  view.panX = mx - size.width / 2 - (mx - size.width / 2 - view.panX) * applied
  view.panY = my - size.height / 2 - (my - size.height / 2 - view.panY) * applied
  view.zoom = zoom
  draw()
}

function onWheel(event: WheelEvent) {
  if (!event.ctrlKey && !event.metaKey) return // a trackpad pinch arrives as ctrl + wheel
  event.preventDefault()
  zoomAt(...local(event), Math.exp(-event.deltaY * 0.0025))
}

function zoomButton(factor: number) {
  zoomAt(size.width / 2, size.height / 2, factor)
}

function onMouseDown(event: MouseEvent) {
  drag = { x: event.clientX, y: event.clientY, moved: false }
}

let frame = 0
function onMouseMove(event: MouseEvent) {
  if (drag) {
    const dx = event.clientX - drag.x
    const dy = event.clientY - drag.y
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true
    view.panX += dx
    view.panY += dy
    drag.x = event.clientX
    drag.y = event.clientY
    draw()
    return
  }
  cancelAnimationFrame(frame)
  frame = requestAnimationFrame(() => {
    const [mx, my] = local(event)
    const index = nearest(mx, my)
    hover.value = index < 0 ? null : { index, left: mx, top: my }
  })
}

function onMouseUp(event: MouseEvent) {
  const wasDrag = drag?.moved
  drag = null
  if (wasDrag) return
  const index = nearest(...local(event))
  if (index >= 0) emit('pick', props.points.ids[index])
}

function resetView() {
  Object.assign(view, { zoom: 1, panX: 0, panY: 0 })
  draw()
}

let observer: ResizeObserver | undefined
onMounted(() => {
  observer = new ResizeObserver(([entry]) => {
    const ratio = window.devicePixelRatio || 1
    size = { width: entry.contentRect.width, height: entry.contentRect.height }
    canvas.value!.width = size.width * ratio
    canvas.value!.height = size.height * ratio
    draw()
  })
  observer.observe(host.value!)
})
onBeforeUnmount(() => observer?.disconnect())

watch(() => [props.points, props.clusterId, props.selected, props.clusterLabel, isDark.value], draw)
watch(() => props.points, resetView)
</script>

<template>
  <div ref="host" class="map">
    <canvas
      ref="canvas"
      role="img"
      :aria-label="`Map of ${points.ids.length} posts; the selected cluster is highlighted`"
      @wheel="onWheel"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
      @mouseleave="hover = null; drag = null"
    ></canvas>
    <div
      v-if="hover"
      class="tooltip"
      :style="{ left: `${hover.left + 12}px`, top: `${hover.top + 12}px` }"
    >
      <span class="mono">{{ points.ids[hover.index] }}</span>
      <span>cluster {{ points.cluster[hover.index] }}</span>
      <span v-if="points.newsgroup" class="muted">{{ points.newsgroup[hover.index] }}</span>
    </div>
    <div class="controls">
      <md-icon-button aria-label="Zoom in" title="Zoom in (or Ctrl/⌘ + scroll)" @click="zoomButton(1.5)">
        <md-icon>add</md-icon>
      </md-icon-button>
      <md-icon-button aria-label="Zoom out" title="Zoom out" @click="zoomButton(1 / 1.5)">
        <md-icon>remove</md-icon>
      </md-icon-button>
      <md-icon-button aria-label="Reset zoom" title="Show the whole map" @click="resetView">
        <md-icon>fit_screen</md-icon>
      </md-icon-button>
    </div>
  </div>
</template>

<style scoped>
.map {
  position: relative;
  height: 520px;
  border-radius: var(--md-sys-shape-corner-medium);
  background: var(--md-sys-color-surface-container-lowest);
  overflow: hidden;
}
canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
canvas:active { cursor: grabbing; }
.tooltip {
  position: absolute;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 10px;
  border-radius: var(--md-sys-shape-corner-extra-small);
  background: var(--md-sys-color-inverse-surface);
  color: var(--md-sys-color-inverse-on-surface);
  font: var(--md-sys-typescale-body-small);
}
.tooltip .muted { color: var(--md-sys-color-inverse-on-surface); opacity: 0.8; }
.controls {
  position: absolute;
  right: 8px;
  top: 8px;
  display: flex;
  flex-direction: column;
  border-radius: var(--md-sys-shape-corner-full);
  background: var(--md-sys-color-surface-container-high);
}
@media (max-width: 599px) { .map { height: 380px; } }
</style>
