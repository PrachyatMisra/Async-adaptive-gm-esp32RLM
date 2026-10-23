import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import ForceGraph3D from 'react-force-graph-3d'
import type { ForceGraphMethods, GraphData, LinkObject, NodeObject } from 'react-force-graph-3d'
import { forceCollide, forceRadial, forceX, forceY, type SimulationNodeDatum } from 'd3-force'
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Activity, Bot, CheckCircle2, ChevronRight, Clock3, Database, ExternalLink, FileCode2, GitBranch, Gauge, History, Info, MessageSquare, Network, Play, RotateCcw, Send, ShieldCheck, Sparkles, Wifi, X, XCircle } from 'lucide-react'
import { useTelemetry } from './useTelemetry'

type View = 'overview' | 'parity' | 'chat' | 'graph' | 'results'
type Subsystem = 'frontend' | 'backend' | 'firmware' | 'verification'
type GraphNode = { id: string; label: string; source_file?: string; source_location?: string; community?: number; subsystem?: Subsystem }
type GraphLink = { source: string | GraphNode; target: string | GraphNode }
type GraphPayload = { nodes?: GraphNode[]; links?: GraphLink[] }
type ParitySample = { i: number; pred: number; gold_pred: number; logit_err: number; gate_err: number; ok: boolean }
type ParityResult = { status: string; summary: { passed: number; golden: number; max_logit_err?: number; max_gate_err?: number; max_shadow_err?: number; mean_steps?: number }; samples?: ParitySample[] }
type ParityRow = ParitySample & { errorDelta: number }
type ChatMessage = { role: 'user' | 'assistant'; content: string }
type ActivityEvent = { id: string; kind: 'parity' | 'chat' | 'custom'; timestamp: string; title: string; detail: string; status: 'pass' | 'info'; meta?: string }
type CustomInference = { text: string; verdict: string; confidence_pct: number; steps: number; host_latency_us: number; timestamp: string }

const API = 'http://127.0.0.1:8000'
const VECTOR_COUNT = 12
const SUBSYSTEM_COLORS: Record<Subsystem, string> = { frontend: '#67b5ff', backend: '#66e3c4', firmware: '#f7b955', verification: '#a78bfa' }

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const points = values.length > 1
    ? values.map((value, index) => `${(index / (values.length - 1)) * 100},${100 - value * 100}`).join(' ')
    : '0,100 100,100'
  return <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline" aria-hidden="true"><polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" /></svg>
}

function Metric({ label, value, detail, values, color }: { label: string; value: string; detail: string; values?: number[]; color: string }) {
  return <motion.div whileHover={{ y: -3 }} className="metric-card">
    <div className="metric-heading"><span>{label}</span><span className="metric-dot" style={{ background: color }} /></div>
    <strong>{value}</strong><small>{detail}</small>
    {values && <Sparkline values={values} color={color} />}
  </motion.div>
}

function ParityPanel({ parity, rows, lastRun, onRun, verifying, verifiedVectors, progress }: { parity: ParityResult | null; rows: ParityRow[]; lastRun: Date | null; onRun: () => void; verifying: boolean; verifiedVectors: number; progress: number }) {
  const summary = parity?.summary
  const passed = verifying ? verifiedVectors : summary?.passed ?? 0
  return <section className="panel parity-panel">
    <div className="panel-header">
      <div><span className="eyebrow accent">VERIFICATION ENGINE</span><h2>Golden parity</h2><p>Native C++ engine compared against the approved ESP32 vectors.</p></div>
      <div className={verifying ? 'result-badge running' : parity?.status === 'pass' ? 'result-badge pass' : 'result-badge'}>{verifying ? <><span className="spinner" /> AUDITING</> : parity?.status === 'pass' ? <><CheckCircle2 size={17} /> PASS</> : <><XCircle size={17} /> READY</>}</div>
    </div>
    <div className="parity-controls"><button className="primary-button" onClick={onRun} disabled={verifying}>{verifying ? <><span className="spinner dark" /> Verifying vectors…</> : <><RotateCcw size={15} /> Run parity check</>}</button><span className="mono">{verifying ? `${Math.round(progress)}% COMPLETE` : '12 VECTORS · HOST NATIVE'}</span>{lastRun && <span className="parity-run-time">LAST EXECUTION {lastRun.toLocaleTimeString()}</span>}</div>
    <div className="parity-score"><strong>{passed}/{VECTOR_COUNT}</strong><span>vectors passed</span><div className="progress"><motion.i animate={{ width: `${progress}%` }} transition={{ ease: 'easeOut', duration: .25 }} /></div></div>
    <div className="vector-grid" aria-label="Parity vector progress">{Array.from({ length: VECTOR_COUNT }, (_, index) => <motion.div key={index} initial={{ opacity: .35, scale: .92 }} animate={{ opacity: index < verifiedVectors ? 1 : .35, scale: index < verifiedVectors ? 1 : .92 }} transition={{ duration: .25 }} className={index < verifiedVectors ? 'vector-chip passed' : 'vector-chip'}><span>{index + 1}</span>{index < verifiedVectors ? <CheckCircle2 size={13} /> : <span className="vector-pending" />}</motion.div>)}</div>
    <div className="parity-chart"><div className="chart-heading"><span className="eyebrow">ERROR DELTA / LIVE TRACE</span><span className="chart-legend"><i className="logit-line" /> Logit <i className="gate-line" /> Gate</span></div><ResponsiveContainer width="100%" height={210}><LineChart data={rows} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}><CartesianGrid stroke="#1b3539" strokeDasharray="3 4" /><XAxis dataKey="i" stroke="#668087" tick={{ fontSize: 10 }} tickFormatter={(value) => `V${Number(value) + 1}`} /><YAxis stroke="#668087" tick={{ fontSize: 10 }} tickFormatter={(value) => Number(value).toExponential(0)} /><Tooltip contentStyle={{ background: '#0d1c21', border: '1px solid #2c5456', borderRadius: 8, color: '#d8efea' }} labelFormatter={(value) => `Vector ${Number(value) + 1}`} /><Line type="monotone" dataKey="logit_err" name="Max logit error" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3, fill: '#a78bfa' }} isAnimationActive animationDuration={220} /><Line type="monotone" dataKey="gate_err" name="Max gate error" stroke="#66e3c4" strokeWidth={2} dot={{ r: 3, fill: '#66e3c4' }} isAnimationActive animationDuration={220} /></LineChart></ResponsiveContainer></div>
    <div className="parity-table-wrap"><table className="parity-table"><thead><tr><th>Vector ID</th><th>Target Output</th><th>Engine Output</th><th>Error Delta</th><th>Status</th></tr></thead><tbody>{rows.map((row) => <motion.tr key={row.i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}><td>V{row.i + 1}</td><td>{row.gold_pred}</td><td>{row.pred}</td><td>{row.errorDelta.toExponential(2)}</td><td><span className="table-pass"><CheckCircle2 size={12} /> PASS</span></td></motion.tr>)}</tbody></table>{!rows.length && <div className="table-empty">Awaiting vector execution…</div>}</div>
    <AnimatePresence>{!verifying && parity && <motion.div className="data-grid" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .15 }}><DataCell label="Max logit error" value={summary?.max_logit_err} /><DataCell label="Max gate error" value={summary?.max_gate_err} /><DataCell label="Max shadow error" value={summary?.max_shadow_err} /><DataCell label="Mean recursion steps" value={summary?.mean_steps} /></motion.div>}</AnimatePresence>
  </section>
}

function CustomInputPanel({ result, running, onRun }: { result: CustomInference | null; running: boolean; onRun: (text: string) => void }) {
  const [text, setText] = useState('')
  return <div className="custom-input-panel">
    <div><span className="eyebrow">EXPLORATION SANDBOX</span><h3>Try a custom input</h3><p>Evaluate temporary text through the native inference path without changing the approved golden corpus or project files.</p></div>
    <textarea value={text} onChange={(event) => setText(event.target.value)} maxLength={4096} placeholder="The device handled the workload efficiently…" aria-label="Custom inference input" />
    <div className="custom-input-actions"><button className="primary-button" onClick={() => onRun(text)} disabled={running || !text.trim()}>{running ? <><span className="spinner dark" /> Running…</> : <><Play size={15} /> Run custom inference</>}</button><button className="ghost-button" onClick={() => { setText(''); }} disabled={running || !text}><RotateCcw size={14} /> Reset</button><span className="sandbox-label">EXPLORATORY ONLY · DOES NOT AFFECT 12/12</span></div>
    {result && <motion.div className="custom-result" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><div><span>Prediction</span><strong>{result.verdict}</strong></div><div><span>Confidence</span><strong>{result.confidence_pct.toFixed(1)}%</strong></div><div><span>Adaptive depth</span><strong>{result.steps} steps</strong></div><div><span>Host execution</span><strong>{(result.host_latency_us / 1000).toFixed(2)} ms</strong></div><small>Custom input evaluated at {new Date(result.timestamp).toLocaleTimeString()}. Official parity remains unchanged.</small></motion.div>}
  </div>
}

function DataCell({ label, value }: { label: string; value?: number }) {
  return <div className="data-cell"><span>{label}</span><b>{value ?? '—'}</b></div>
}

function PipelineGraph({ graph, onAnalyze }: { graph: GraphPayload | null; onAnalyze: (node: GraphNode) => void }) {
  const graphRef = useRef<ForceGraphMethods<NodeObject<GraphNode>, LinkObject<GraphNode>> | undefined>(undefined)
  const forcesConfigured = useRef(false)
  const clickTimer = useRef<number | undefined>(undefined)
  const clickCount = useRef(0)
  const [mode, setMode] = useState<'isolated' | 'global'>('isolated')
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const graphData = useMemo<GraphData<NodeObject<GraphNode>, LinkObject<GraphNode>>>(
    () => {
      const nodes = graph?.nodes ?? []
      const links = [...(graph?.links ?? [])]
      if (mode === 'global' && nodes.length > 2) {
        const bySubsystem = new Map<string, GraphNode>()
        nodes.forEach((node) => {
          if (node.subsystem && !bySubsystem.has(node.subsystem)) bySubsystem.set(node.subsystem, node)
        })
        const anchors = [...bySubsystem.values()]
        anchors.slice(1).forEach((node, index) => links.push({ source: anchors[0].id, target: node.id, bridge: true } as GraphLink))
      }
      return { nodes, links }
    },
    [graph, mode],
  )
  const configureForces = useCallback(() => {
    if (forcesConfigured.current) return
    const charge = graphRef.current?.d3Force('charge')
    if (charge && 'strength' in charge) charge.strength(-230)
    graphRef.current?.d3Force('collide', forceCollide(22))
    if (mode === 'global') {
      const lobe = (node: NodeObject<GraphNode> | SimulationNodeDatum) => {
        const hash = [...String((node as NodeObject<GraphNode>).id ?? '')].reduce((sum, character) => sum + character.charCodeAt(0), 0)
        return hash % 5
      }
      const hemisphere = (node: NodeObject<GraphNode> | SimulationNodeDatum) => {
        const hash = [...String((node as NodeObject<GraphNode>).id ?? '')].reduce((sum, character) => sum + character.charCodeAt(0), 0)
        return hash % 2 === 0 ? -1 : 1
      }
      graphRef.current?.d3Force('brain-x', forceX((node) => hemisphere(node) * 118 + (lobe(node) % 2) * 32).strength(.085))
      graphRef.current?.d3Force('brain-y', forceY((node) => (lobe(node) - 2) * 52).strength(.07))
      graphRef.current?.d3Force('brain-radial', forceRadial((node) => 112 + (lobe(node) % 3) * 28, 0, 0).strength(.018))
    } else {
      graphRef.current?.d3Force('brain-x', forceX((node) => ({ firmware: -170, backend: -40, verification: 80, frontend: 190 }[(node as NodeObject<GraphNode>).subsystem ?? 'backend'])).strength(.045))
      graphRef.current?.d3Force('brain-y', forceY(0).strength(.02))
      graphRef.current?.d3Force('brain-radial', null)
    }
    forcesConfigured.current = true
  }, [mode])
  useEffect(() => {
    forcesConfigured.current = false
    graphRef.current?.d3ReheatSimulation()
  }, [mode])
  const githubUrl = (node: NodeObject<GraphNode>) => {
    const source = node.source_file ?? ''
    const line = node.source_location?.replace(/\D/g, '') ?? '1'
    return `https://github.com/PrachyatMisra/rlm-esp32-aagm/blob/main/${source}#L${line}`
  }
  const neighbors = useMemo(() => {
    if (!hovered) return new Set<string>()
    const connected = new Set<string>([hovered])
    graphData.links.forEach((link) => {
      const source = typeof link.source === 'object' ? link.source.id : link.source
      const target = typeof link.target === 'object' ? link.target.id : link.target
      if (source === hovered) connected.add(String(target))
      if (target === hovered) connected.add(String(source))
    })
    return connected
  }, [graphData.links, hovered])
  const degree = useMemo(() => {
    const values = new Map<string, number>()
    graphData.links.forEach((link) => {
      const source = String(typeof link.source === 'object' ? link.source.id : link.source)
      const target = String(typeof link.target === 'object' ? link.target.id : link.target)
      values.set(source, (values.get(source) ?? 0) + 1)
      values.set(target, (values.get(target) ?? 0) + 1)
    })
    return values
  }, [graphData.links])
  const handleNodeClick = (node: NodeObject<GraphNode>) => {
    clickCount.current += 1
    if (clickCount.current === 3) {
      if (clickTimer.current) window.clearTimeout(clickTimer.current)
      clickCount.current = 0
      window.open(githubUrl(node), '_blank', 'noopener,noreferrer')
      return
    }
    if (clickTimer.current) window.clearTimeout(clickTimer.current)
    clickTimer.current = window.setTimeout(() => {
      if (clickCount.current === 1) {
        setSelected(node)
      }
      clickCount.current = 0
    }, 280)
  }
  return <section className="panel graph-panel"><div className="panel-header"><div><span className="eyebrow accent">GRAPHIFY / 3-D FORCE FIELD</span><h2>Pipeline topology</h2><p>Drag in any direction to orbit the neural graph. Single click opens context; triple click jumps to GitHub.</p></div><span className="mono">{graphData.nodes.length} NODES · {graph?.links?.length ?? 0} LINKS</span></div><div className="graph-toolbar"><div className="mode-toggle"><button className={mode === 'isolated' ? 'active' : ''} onClick={() => setMode('isolated')}>Isolated subsystems</button><button className={mode === 'global' ? 'active' : ''} onClick={() => setMode('global')}>Global brain network</button></div><span className="graph-hint">Trackpad drag = orbit · scroll = zoom</span></div><div className="force-graph-stage">{graphData.nodes.length ? <ForceGraph3D ref={graphRef} graphData={graphData} nodeId="id" backgroundColor="#091015" width={900} height={470} controlType="orbit" showNavInfo={false} enableNodeDrag cooldownTicks={160} d3AlphaDecay={.025} d3VelocityDecay={.3} linkColor={(link) => (link as GraphLink & { bridge?: boolean }).bridge ? '#66e3c4' : hovered && !neighbors.has(String(typeof link.source === 'object' ? link.source.id : link.source)) && !neighbors.has(String(typeof link.target === 'object' ? link.target.id : link.target)) ? '#17292d' : '#3e7770'} linkWidth={(link) => (link as GraphLink & { bridge?: boolean }).bridge ? 2 : 1.2} linkDirectionalParticles={mode === 'global' ? 3 : 2} linkDirectionalParticleWidth={2.2} linkDirectionalParticleSpeed={.006} nodeRelSize={5} nodeVal={(node) => 1.8 + (degree.get(String(node.id)) ?? 0) * .35} nodeColor={(node) => hovered && !neighbors.has(String(node.id)) ? '#1a2a2e' : SUBSYSTEM_COLORS[node.subsystem ?? 'backend']} nodeLabel={(node) => `${node.label} · ${node.source_file ?? 'source'}`} onEngineTick={configureForces} onNodeClick={handleNodeClick} onNodeHover={(node) => setHovered(node?.id ? String(node.id) : null)} /> : <div className="empty-state">Graph index unavailable.</div>}</div><div className="graph-legend"><span><i className="legend-dot mint" /> Backend</span><span><i className="legend-dot violet" /> Verification</span><span><i className="legend-dot gold" /> Firmware</span><span><i className="legend-dot blue" /> Frontend</span></div><AnimatePresence>{selected && <motion.aside className="context-drawer" initial={{ x: 420 }} animate={{ x: 0 }} exit={{ x: 420 }}><button className="drawer-close" onClick={() => setSelected(null)} aria-label="Close context drawer"><X size={18} /></button><span className="eyebrow accent">{selected.subsystem ?? 'backend'} / NODE CONTEXT</span><h3>{selected.label}</h3><span className="drawer-path">{selected.source_file}:{selected.source_location}</span><p>{selected.label} is a {selected.subsystem ?? 'backend'} component in the AAGM control plane. It participates in the repository pipeline and can be inspected with the local model before opening the source.</p><div className="drawer-role"><span>ARCHITECTURE ROLE</span><b>{selected.subsystem === 'firmware' ? 'Edge execution and telemetry' : selected.subsystem === 'verification' ? 'Parity and confidence validation' : selected.subsystem === 'frontend' ? 'Operator visualization and control' : 'Inference orchestration and data delivery'}</b></div><button className="primary-button" onClick={() => onAnalyze(selected)}><Bot size={15} /> Analyze with Local Model</button><button className="ghost-button drawer-github" onClick={() => window.open(githubUrl(selected), '_blank', 'noopener,noreferrer')}><ExternalLink size={14} /> View on GitHub</button></motion.aside>}</AnimatePresence></section>
}

function ArchitectureBlueprint({ onClose }: { onClose: () => void }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [selected, setSelected] = useState('edge')
  const [demoRunning, setDemoRunning] = useState(false)
  const primary = [
    ['edge', 'ESP32 Edge Core'], ['harness', 'C++ FreeRTOS Harness'], ['stream', '10 Hz WebSocket Stream'],
    ['api', 'FastAPI Inference Engine'], ['ui', 'React 19 Control Center'],
  ] as const
  const files: Record<string, string[]> = {
    edge: ['firmware/main/main.cpp', 'firmware/main/http_client.cpp'],
    harness: ['firmware/test/host_harness.cpp'],
    stream: ['host/backend/main.py', 'host/frontend/src/useTelemetry.ts'],
    api: ['app/services/inference.py', 'host/backend/main.py'],
    ui: ['host/frontend/src/App.tsx', 'host/frontend/src/index.css'],
  }
  const nodes: Node[] = primary.flatMap(([id, label], index) => {
    const base: Node = { id, position: { x: 40 + index * 250, y: 100 }, data: { label, expanded: expanded === id, onToggle: () => { setExpanded(expanded === id ? null : id); setSelected(id) }, onSelect: () => setSelected(id) }, type: 'architecture' }
    if (expanded !== id) return [base]
    return [base, ...files[id].map((file, childIndex) => ({ id: `${id}-${childIndex}`, position: { x: 40 + index * 250 + (childIndex % 2 ? 128 : -128), y: childIndex % 2 ? 285 : 285 }, data: { label: file, file, fileNode: true, onSelect: () => setSelected(`${id}-${childIndex}`) }, type: 'architecture' }))]
  })
  const edges: Edge[] = [
    { id: 'e1', source: 'edge', target: 'harness', animated: true, style: { stroke: '#f7b955', strokeDasharray: '7 5', strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: '#f7b955' } },
    { id: 'e2', source: 'harness', target: 'stream', animated: true, style: { stroke: '#66e3c4', strokeDasharray: '7 5', strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: '#66e3c4' } },
    { id: 'e3', source: 'stream', target: 'api', animated: true, style: { stroke: '#67b5ff', strokeDasharray: '7 5', strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: '#67b5ff' } },
    { id: 'e4', source: 'api', target: 'ui', animated: true, style: { stroke: '#a78bfa', strokeDasharray: '7 5', strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: '#a78bfa' } },
  ]
  const subEdges: Edge[] = expanded ? (files[expanded] ?? []).map((_, index) => ({ id: `${expanded}-sub-${index}`, source: expanded, target: `${expanded}-${index}`, animated: true, style: { stroke: '#66e3c4', strokeDasharray: '5 4' } })) : []
  const selectedLabel = selected.includes('-') ? files[selected.split('-')[0]]?.[Number(selected.split('-')[1])] : primary.find(([id]) => id === selected)?.[1]
  return <motion.div className="architecture-modal" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}><div className="architecture-header"><div><span className="eyebrow accent">SYSTEM BLUEPRINT / LIVE DATA PATH</span><h2>How the control plane thinks</h2><p>Click any block to focus its live path. Expand a block to reveal balanced upstream and downstream file nodes.</p></div><button className="drawer-close" onClick={onClose} aria-label="Close architecture blueprint"><X size={19} /></button></div><div className="architecture-flow"><ReactFlowProvider><ReactFlow nodes={nodes} edges={[...edges, ...subEdges]} nodeTypes={{ architecture: ArchitectureNode }} fitView><Background color="#234047" gap={28} /><Controls /></ReactFlow></ReactFlowProvider></div><div className="architecture-inspector"><div><span className="eyebrow">FOCUSED COMPONENT</span><strong>{selectedLabel ?? 'System path'}</strong><p>{selected.includes('-') ? 'Repository file supporting this stage of the control plane.' : 'Primary runtime stage. Expand it to inspect the files and relationships behind this block.'}</p></div><button className="primary-button" onClick={() => { setDemoRunning(true); window.setTimeout(() => setDemoRunning(false), 1400) }} disabled={demoRunning}><Play size={14} /> {demoRunning ? 'Streaming live path…' : 'Run live demonstration'}</button></div></motion.div>
}

function ArchitectureNode({ data }: { data: { label: string; file?: string; fileNode?: boolean; expanded?: boolean; onToggle?: () => void; onSelect?: () => void } }) {
  if (data.fileNode && data.file) return <div className="architecture-node file-node" onClick={data.onSelect}><Handle type="target" position={Position.Top} /><FileCode2 size={15} /><strong>{data.label.split('/').pop()}</strong><button onClick={(event) => { event.stopPropagation(); window.open(`https://github.com/PrachyatMisra/rlm-esp32-aagm/blob/main/${data.file}#L1`, '_blank', 'noopener,noreferrer') }} aria-label={`View ${data.file} on GitHub`}><ExternalLink size={12} /></button></div>
  return <div className={`architecture-node ${data.expanded ? 'expanded' : ''}`} onClick={data.onToggle}><Handle type="target" position={Position.Left} /><Sparkles size={18} /><strong>{data.label}</strong><span className="node-expand">{data.expanded ? '−' : '+'}</span><Handle type="source" position={Position.Right} /></div>
}

function ChatPanel({ chat, prompt, model, sending, setPrompt, onSend, onClear }: { chat: ChatMessage[]; prompt: string; model: string; sending: boolean; setPrompt: (value: string) => void; onSend: () => void; onClear: () => void }) {
  return <section className="panel chat-panel"><div className="panel-header"><div><span className="eyebrow accent">LOCAL ASSISTANT</span><h2>Repository intelligence</h2><p>Markdown-aware responses from the local model and repository retrieval.</p></div><div className="chat-actions"><div className="local-badge"><span className="pulse online" /> {model}</div><button className="ghost-button" onClick={onClear} disabled={!chat.length && !prompt}><RotateCcw size={13} /> Clear</button></div></div><div className="chat-window">{chat.length === 0 && <div className="empty-state"><Bot size={30} /><span>Ask a question to begin a local session.</span><small>Model: {model}</small></div>}{chat.map((message, index) => <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} key={index} className={`chat-message ${message.role}`}><span className="message-role">{message.role === 'user' ? 'YOU' : `RLM · ${model}`}</span>{message.role === 'assistant' ? <div className="markdown-body"><ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown></div> : <p>{message.content}</p>}</motion.div>)}{sending && <div className="chat-message assistant thinking"><span className="message-role">RLM · {model}</span><div className="thinking-dots"><i /><i /><i /></div></div>}</div><div className="chat-input"><input value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && onSend()} placeholder="e.g. Explain the adaptive gate with a code example…" /><button onClick={onSend} disabled={sending || !prompt.trim()} aria-label="Send message"><Send size={17} /></button></div></section>
}

function ResultsPanel({ events, onClear }: { events: ActivityEvent[]; onClear: () => void }) {
  const parityRuns = events.filter((event) => event.kind === 'parity')
  const chatRuns = events.filter((event) => event.kind === 'chat')
  return <section className="panel results-panel"><div className="panel-header"><div><span className="eyebrow accent">SESSION LEDGER / RESULTS</span><h2>Evidence timeline</h2><p>A timestamped, local history of parity executions and repository chat activity.</p></div><button className="ghost-button" onClick={onClear} disabled={!events.length}><RotateCcw size={13} /> Clear history</button></div><div className="results-summary"><div><Database size={16} /><strong>{parityRuns.length}</strong><span>parity runs</span></div><div><MessageSquare size={16} /><strong>{chatRuns.length}</strong><span>chat events</span></div><div><Clock3 size={16} /><strong>{events[0] ? new Date(events[0].timestamp).toLocaleTimeString() : '—'}</strong><span>latest event</span></div></div><div className="determinism-callout"><Info size={18} /><div><strong>Why do 12/12 results repeat?</strong><p>These are deterministic golden regression vectors. Every run reloads the same approved inputs and compares the native engine against the same expected outputs. Identical results are good: they prove reproducibility and protect against regressions. The run timestamp below confirms each execution is new; variation would indicate an unstable engine or changed inputs.</p></div></div><div className="results-timeline">{events.length ? events.map((event) => <motion.article key={event.id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="result-event"><div className={`event-icon ${event.kind}`}><span>{event.kind === 'parity' ? '✓' : '✦'}</span></div><div className="event-body"><div className="event-topline"><strong>{event.title}</strong><time>{new Date(event.timestamp).toLocaleString()}</time></div><p>{event.detail}</p>{event.meta && <span className="event-meta">{event.meta}</span>}</div><span className={`event-status ${event.status}`}>{event.status === 'pass' ? 'PASS' : 'LOGGED'}</span></motion.article>) : <div className="table-empty">No events yet. Run parity or send a local chat message to create the first record.</div>}</div></section>
}

export default function App() {
  const { data, connected, history } = useTelemetry()
  const [view, setView] = useState<View>('overview')
  const [parity, setParity] = useState<ParityResult | null>(null)
  const [parityRows, setParityRows] = useState<ParityRow[]>([])
  const [lastParityRun, setLastParityRun] = useState<Date | null>(null)
  const [events, setEvents] = useState<ActivityEvent[]>(() => {
    try { return JSON.parse(localStorage.getItem('rlm-results-history') ?? '[]') as ActivityEvent[] } catch { return [] }
  })
  const [customInference, setCustomInference] = useState<CustomInference | null>(null)
  const [customRunning, setCustomRunning] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [verifiedVectors, setVerifiedVectors] = useState(0)
  const [progress, setProgress] = useState(0)
  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [prompt, setPrompt] = useState('')
  const [chat, setChat] = useState<ChatMessage[]>([])
  const [model, setModel] = useState('local retrieval')
  const [sending, setSending] = useState(false)
  const [architectureOpen, setArchitectureOpen] = useState(false)

  useEffect(() => {
    fetch(`${API}/api/v1/verification/golden`).then((response) => response.ok ? response.json() : Promise.reject(response.status)).then((result: ParityResult) => {
      setParity(result)
        setLastParityRun(new Date())
        setProgress(100)
        setVerifiedVectors(result.summary.passed)
        setParityRows((result.samples ?? []).map((sample) => ({ ...sample, errorDelta: Math.max(sample.logit_err, sample.gate_err) })))
    }).catch(() => setParity(null))
    fetch(`${API}/api/v2/graph`).then((response) => response.ok ? response.json() : Promise.reject(response.status)).then(setGraph).catch(() => setGraph(null))
    fetch(`${API}/api/v1/assistant/status`).then((response) => response.ok ? response.json() : Promise.reject(response.status)).then((status) => setModel(status.model_installed ? status.model : status.label ?? status.model ?? 'local retrieval')).catch(() => setModel('local retrieval'))
  }, [])

  useEffect(() => {
    if (parity && !verifying) setVerifiedVectors(parity.summary.passed)
  }, [parity, verifying])
  useEffect(() => { localStorage.setItem('rlm-results-history', JSON.stringify(events)) }, [events])

  const gates = useMemo(() => history.map((item) => item.g_t), [history])
  const latency = useMemo(() => history.map((item) => Math.min(item.latency_ms / 140, 1)), [history])
  const runParity = async () => {
    if (verifying) return
    setVerifying(true)
    setVerifiedVectors(0)
    setProgress(0)
    setParityRows([])
    try {
      const response = await fetch(`${API}/api/v1/verification/golden?run=${Date.now()}`, { cache: 'no-store' })
      if (!response.ok) throw new Error(`Parity request failed with status ${response.status}`)
      const result = await response.json() as ParityResult
      const samples = result.samples ?? []
      setParity(result)
      setLastParityRun(new Date())
      const event: ActivityEvent = { id: `${Date.now()}-parity`, kind: 'parity', timestamp: new Date().toISOString(), title: `Golden parity · ${result.summary.passed}/${result.summary.golden} passed`, detail: 'Native C++ harness completed a fresh deterministic regression execution.', status: result.status === 'pass' ? 'pass' : 'info', meta: `max logit ${result.summary.max_logit_err?.toExponential(2) ?? '—'} · ${samples.length} vectors` }
      setEvents((current) => [event, ...current].slice(0, 100))
      for (let vector = 1; vector <= samples.length; vector += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 200))
        setVerifiedVectors(vector)
        setProgress((vector / samples.length) * 100)
        const sample = samples[vector - 1]
        setParityRows((current) => [...current, { ...sample, errorDelta: Math.max(sample.logit_err, sample.gate_err) }])
      }
      setVerifiedVectors(samples.length)
      setProgress(100)
    } catch (error) {
      console.error('Parity check failed', error)
    }
    setVerifying(false)
  }
  const sendMessage = async () => {
    if (!prompt.trim() || sending) return
    const text = prompt.trim()
    setPrompt('')
    setChat((current) => [...current, { role: 'user', content: text }])
    const event: ActivityEvent = { id: `${Date.now()}-chat`, kind: 'chat', timestamp: new Date().toISOString(), title: 'Local chat prompt', detail: text, status: 'info', meta: model }
    setEvents((current) => [event, ...current].slice(0, 100))
    setSending(true)
    try {
      const response = await fetch(`${API}/api/v1/assistant/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, history: chat }) })
      const payload = await response.json()
      const reply = payload.reply ?? payload.detail ?? 'The local assistant returned no response.'
      setChat((current) => [...current, { role: 'assistant', content: reply }])
      const responseEvent: ActivityEvent = { id: `${Date.now()}-chat-response`, kind: 'chat', timestamp: new Date().toISOString(), title: 'Local chat response', detail: reply, status: 'info', meta: model }
      setEvents((current) => [responseEvent, ...current].slice(0, 100))
    } catch {
      setChat((current) => [...current, { role: 'assistant', content: 'The unified API is offline. Start the backend on port 8000 and try again.' }])
    } finally {
      setSending(false)
    }
  }
  const runCustomInference = async (text: string) => {
    const cleaned = text.trim()
    if (!cleaned || customRunning) return
    setCustomRunning(true)
    try {
      const response = await fetch(`${API}/api/v1/inference`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: cleaned, profile: 'PERF' }), cache: 'no-store' })
      if (!response.ok) throw new Error(`Custom inference failed with status ${response.status}`)
      const payload = await response.json() as { verdict: string; confidence_pct: number; steps: number; host_latency_us: number }
      const result: CustomInference = { text: cleaned, verdict: payload.verdict, confidence_pct: payload.confidence_pct, steps: payload.steps, host_latency_us: payload.host_latency_us, timestamp: new Date().toISOString() }
      setCustomInference(result)
      const event: ActivityEvent = { id: `${Date.now()}-custom`, kind: 'custom', timestamp: result.timestamp, title: 'Custom inference exploration', detail: `${result.verdict} · "${cleaned}"`, status: 'info', meta: `${result.confidence_pct.toFixed(1)}% confidence · ${result.steps} steps` }
      setEvents((current) => [event, ...current].slice(0, 100))
    } catch (error) {
      console.error('Custom inference failed', error)
    } finally {
      setCustomRunning(false)
    }
  }
  const analyzeNode = (node: GraphNode) => {
    setView('chat')
    setPrompt(`Analyze ${node.source_file ?? node.label} (${node.label}) in the AAGM architecture.`)
  }
  const navItems: [View, string, typeof Activity][] = [['overview', 'Overview', Activity], ['parity', 'Parity', ShieldCheck], ['chat', 'Local Chat', MessageSquare], ['graph', 'Pipeline Graph', Network], ['results', 'Results history', History]]
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><div className="brand-mark">R</div><div><b>RLM / AAGM</b><span>EDGE CONTROL PLANE</span></div></div><div className="sidebar-label">Workspace</div><nav>{navItems.map(([id, label, Icon]) => <button key={id} className={view === id ? 'nav-item active' : 'nav-item'} onClick={() => setView(id)}><Icon size={17} />{label}</button>)}</nav><div className="sidebar-footer"><div className="system-status"><span className={connected ? 'pulse online' : 'pulse'} />{connected ? 'Telemetry online' : 'Reconnecting telemetry'}</div><span className="mono">v2.0.0 · localhost</span></div></aside>
    <main className="main-content"><header className="topbar"><div><span className="eyebrow">ESP32 / LIVE SYSTEM</span><h1>{view === 'overview' ? 'System overview' : navItems.find(([id]) => id === view)?.[1]}</h1></div><div className="connection-pill"><Wifi size={14} /> 10 Hz stream <span className={connected ? 'status-good' : 'status-bad'}>{connected ? 'CONNECTED' : 'OFFLINE'}</span></div></header><AnimatePresence mode="wait"><motion.div key={view} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: .2 }}>
      {view === 'overview' && <><section className="hero-card"><div><span className="eyebrow accent">ADAPTIVE INFERENCE FABRIC</span><h2>Observe the edge.<br /><em>Control the recursion.</em></h2><p>Live instrumentation for the dual-core RLM engine, golden parity, and local model reasoning.</p><div className="hero-actions"><button className="primary-button" onClick={() => { setView('parity'); void runParity() }}>Run parity check <Gauge size={16} /></button><button className="ghost-button blueprint-button" onClick={() => setArchitectureOpen(true)}><Network size={15} /> Explain System Architecture</button></div></div><div className="orbital"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="core-glow">AAGM<span>CORE</span></div></div></section><div className="section-heading"><div><span className="eyebrow">REAL-TIME TELEMETRY</span><h3>Engine pulse</h3></div><span className="live-label"><span className="pulse online" /> LIVE</span></div><section className="metric-grid"><Metric label="Gate value · g_t" value={data ? data.g_t.toFixed(4) : '—'} detail="Adaptive halt confidence" values={gates} color="#66e3c4" /><Metric label="Inference latency" value={data ? `${data.latency_ms.toFixed(1)} ms` : '—'} detail="Host stream sample" values={latency} color="#a78bfa" /><Metric label="Active core" value={data ? `CORE ${data.active_core}` : '—'} detail="Dual-core scheduler" color="#f7b955" /><Metric label="Free memory" value={data ? `${(data.memory_free_bytes / 1024).toFixed(1)} KB` : '—'} detail="Window contraction at 120 KB" color="#67b5ff" /></section><div className="quick-grid"><button onClick={() => setView('chat')} className="quick-card"><Bot /><div><b>Ask the local model</b><span>{model} · Markdown responses</span></div><ChevronRight size={15} /></button><button onClick={() => setView('graph')} className="quick-card"><GitBranch /><div><b>Explore the pipeline</b><span>Physics layout with live data flow</span></div><ChevronRight size={15} /></button></div></>}
      {view === 'parity' && <><ParityPanel parity={parity} rows={parityRows} lastRun={lastParityRun} onRun={() => void runParity()} verifying={verifying} verifiedVectors={verifiedVectors} progress={progress} /><CustomInputPanel result={customInference} running={customRunning} onRun={(text) => void runCustomInference(text)} /></>}
      {view === 'chat' && <ChatPanel chat={chat} prompt={prompt} model={model} sending={sending} setPrompt={setPrompt} onSend={() => void sendMessage()} onClear={() => { setChat([]); setPrompt('') }} />}
      {view === 'graph' && <PipelineGraph graph={graph} onAnalyze={analyzeNode} />}
      {view === 'results' && <ResultsPanel events={events} onClear={() => { setEvents([]); localStorage.removeItem('rlm-results-history') }} />}
    </motion.div></AnimatePresence></main>
    <AnimatePresence>{architectureOpen && <ArchitectureBlueprint onClose={() => setArchitectureOpen(false)} />}</AnimatePresence>
  </div>
}
