import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {api, pick, type Json} from './api';
import './style.css';

type View = 'Conversations' | 'Resources' | 'Skills' | 'Training' | 'Models';
const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const json = (value: unknown) => JSON.stringify(value, null, 2);

function App() {
  const [view, setView] = useState<View>('Conversations');
  const [status, setStatus] = useState<Json>(null);
  const [error, setError] = useState('');
  const [conversations, setConversations] = useState<Json[]>([]);
  const [conversation, setConversation] = useState('');
  const [search, setSearch] = useState('');
  const [messages, setMessages] = useState<Json[]>([]);
  const [graphs, setGraphs] = useState<Json[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [skills, setSkills] = useState<Json[]>([]);
  const [checkpoints, setCheckpoints] = useState<Json[]>([]);
  const [jobs, setJobs] = useState<Json[]>([]);
  const [text, setText] = useState('');
  const [resourcePath, setResourcePath] = useState('');
  const [query, setQuery] = useState('');
  const [evidence, setEvidence] = useState<Json[]>([]);
  const [threshold, setThreshold] = useState(100);
  const [observation, setObservation] = useState(60);
  const [epochs, setEpochs] = useState(80);
  const [graphFilter, setGraphFilter] = useState('');
  const [importing, setImporting] = useState<Json>(null);
  const [inspecting, setInspecting] = useState<Json>(null);
  const [proposal, setProposal] = useState<Json>(null);
  const [combine, setCombine] = useState(false);
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState('');
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [jobDetail, setJobDetail] = useState<Json>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const setupStarted = useRef(false);
  const currentConversation = useRef(conversation);
  const liveSeq = useRef<Record<string, number>>({});
  currentConversation.current = conversation;

  async function action(fn: () => Promise<unknown>) {
    setError(''); setBusy(true);
    try {await fn();} catch (e) {setError(String(e));} finally {setBusy(false);}
  }
  async function refresh() {
    const [s, cs, gs, ss, cps, js] = await Promise.all([
      api('GET', '/v1/status'), api('GET', '/v1/conversations?q='+encodeURIComponent(search)),
      api('GET', '/v1/graphs'), api('GET', '/v1/skills'), api('GET', '/v1/checkpoints'), api('GET', '/v1/jobs')]);
    setStatus(s); setConversations(cs); setGraphs(gs); setSkills(ss); setCheckpoints(cps); setJobs(js);
    setResourcePath(previous => previous || s.resources);
    if (s.first_run && !setupStarted.current) {
      setupStarted.current = true;
      await api('POST', '/v1/setup', {});
    }
  }
  useEffect(() => {void action(refresh);}, []);
  useEffect(() => {
    let cancelled = false;
    void api('GET', '/v1/conversations?q='+encodeURIComponent(search)).then(cs => {if (!cancelled) setConversations(cs);}).catch(e => setError(String(e)));
    return () => {cancelled = true;};
  }, [search]);
  useEffect(() => {
    setMessages([]); setLive('');
    if (!conversation) return;
    let cancelled = false;
    void api('GET', `/v1/conversations/${conversation}/messages`).then(ms => {if (!cancelled) setMessages(ms);}).catch(e => setError(String(e)));
    return () => {cancelled = true;};
  }, [conversation]);
  useEffect(() => {
    let running = false;
    const timer = setInterval(async () => {
      if (running) return;
      running = true;
      try {
        const js = await api('GET', '/v1/jobs'); setJobs(js);
        const chat = js.find((j: Json) => j.kind === `chat:${currentConversation.current}` && !terminal.has(j.status));
        setActiveChat(chat?.id || null);
        if (chat) {
          const tokens = chat.events.filter((e: Json) => e.event === 'token').map((e: Json) => e.text).join('');
          setLive(tokens);
        }
        const completed = js.filter((j: Json) => terminal.has(j.status) && liveSeq.current[j.id] !== j.events.length);
        if (completed.length) {
          for (const j of completed) liveSeq.current[j.id] = j.events.length;
          const s = await api('GET', '/v1/status'); setStatus(s);
          setGraphs(await api('GET', '/v1/graphs'));
          setCheckpoints(await api('GET', '/v1/checkpoints'));
          if (currentConversation.current) {
            setMessages(await api('GET', `/v1/conversations/${currentConversation.current}/messages`));
            setLive('');
          }
        }
      } catch (e) {setError(String(e));} finally {running = false;}
    }, 1200);
    return () => clearInterval(timer);
  }, []);

  async function addConversation() {
    const c = await api('POST', '/v1/conversations', {title: 'New conversation'});
    setConversation(c.id); setSelected([]); setView('Conversations'); await refresh();
  }
  async function send() {
    let id = conversation;
    if (!id) {
      const c = await api('POST', '/v1/conversations', {title: text.slice(0, 70)}); id = c.id; setConversation(id);
    }
    const result = await api('POST', `/v1/conversations/${id}/chat`, {text, graph_ids: selected});
    setActiveChat(result.job_id); setText(''); setLive(''); await refresh();
  }
  async function importPosts() {
    const path = await pick('posts'); if (!path) return;
    const preview = await api('POST', '/v1/posts/preview', {path}); setCombine(false); setImporting({...preview, path});
  }
  async function confirmImport() {
    const result = await api('POST', '/v1/posts/import', {path: importing.path, mapping: importing.mapping, combine});
    setSelected(result.graph_ids); setImporting(null); await refresh();
  }
  async function inspectGraph(id: string) {
    setInspecting(await api('GET', `/v1/graphs/${id}`));
    setProposal(await api('GET', `/v1/graphs/${id}/proposal`));
  }
  async function waitFor(id: string) {
    while (true) {
      const job = await api('GET', `/v1/jobs/${id}`);
      if (terminal.has(job.status)) {
        if (job.status !== 'completed') throw Error(job.error || job.status);
        return job.result;
      }
      await new Promise(r => setTimeout(r, 700));
    }
  }
  const activeJobs = jobs.filter(j => !terminal.has(j.status));
  const latestTraining = jobs.find(j => j.kind === 'training');
  const visibleGraphs = graphs.filter(g => (g.cascade_id+' '+g.source).toLowerCase().includes(graphFilter.toLowerCase())).slice(0, 60);

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><span className="brand-icon">V</span><div>VERITAS<small>Evidence & propagation</small></div></div>
      <div className="workspace-label">LOCAL RESEARCH WORKSPACE</div>
      <nav>{(['Conversations', 'Resources', 'Skills', 'Training', 'Models'] as View[]).map((v, i) =>
        <button key={v} className={view === v ? 'nav active' : 'nav'} onClick={() => setView(v)}><span>{['◌', '▤', '◇', '⌁', '⬡'][i]}</span>{v}</button>)}</nav>
      <div className="sidebar-rule"/>
      <div className="row between"><span className="workspace-label">CONVERSATIONS</span><button className="icon-button" aria-label="New conversation" onClick={() => void action(addConversation)}>＋</button></div>
      <input className="search" placeholder="Search conversations…" value={search} onChange={e => setSearch(e.target.value)}/>
      <div className="conversation-list">{conversations.map(c => <button key={c.id} className={'conversation '+(conversation === c.id ? 'chosen' : '')} onClick={() => {setConversation(c.id); setSelected([]); setView('Conversations');}}>{c.title}</button>)}</div>
      <div className="sidebar-footer"><span className="status-dot"/>Local inference<small>Your posts stay on this device</small><button className="link" onClick={() => void window.veritas.openData()}>Open data directory ↗</button></div>
    </aside>
    <main>
      <header><div><span className="eyebrow">RESEARCH STUDIO</span><h1>{view}</h1></div><div className="row"><span className="pill">Qwen3.5 · 9B</span><span className="pill subtle">{status?.qwen_path ? 'Model available' : 'Setup required'}</span></div></header>
      {error && <div role="alert" className="alert error">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
      {view === 'Conversations' && <div className="conversation-layout">
        <section className="chat-panel">
          <div className="chat-toolbar"><span>{conversations.find(c => c.id === conversation)?.title || 'New conversation'}</span><div className="row">
            {conversation && <><button className="link" onClick={() => setRenaming(conversations.find(c => c.id === conversation)?.title || '')}>Rename</button>
            <button className="link danger" onClick={() => void action(async () => {if (window.confirm('Delete this conversation and its messages?')) {await api('DELETE', `/v1/conversations/${conversation}`); setConversation(''); await refresh();}})}>Delete</button></>}
          </div></div>
          <div className="messages" aria-live="polite">
            {!messages.length && !activeChat && <div className="welcome"><div className="welcome-symbol">◎</div><span className="eyebrow">EVIDENCE MEETS NETWORK SCIENCE</span><h2>Understand the claim.<br/>Explore its potential reach.</h2><p>Check statements against local evidence and analyze how messages may spread. Two questions, one workspace.</p><div className="suggestions"><button onClick={() => setText('Check the factual claims in the attached posts and cite the evidence.')}>Check factual support <span>↗</span></button><button onClick={() => setText('Predict virality for the attached graphs. Explain the observation window and any estimated metadata.')}>Analyze propagation <span>↗</span></button></div></div>}
            {messages.filter(m => m.role !== 'tool').map(m => <article key={m.id} className={'message '+m.role}><div className="message-label">{m.role === 'user' ? 'YOU' : 'VERITAS'}</div><p>{m.content}</p>
              {m.data.claims?.map((c: Json, i: number) => <div className="claim" key={i}><span className={'verdict '+c.verdict}>{c.verdict.replaceAll('_', ' ')}</span><strong>{c.claim}</strong><p>{c.explanation}</p><span className="citation">{c.citations.join(' · ')}</span></div>)}
              {m.data.tools?.filter((t: Json) => t.name === 'graphsage_predict').map((t: Json, i: number) => <div key={i} className="tool-result"><span className="eyebrow">GRAPHSAGE · VIRALITY</span>{Array.isArray(t.result) ? t.result.map((p: Json) => <div key={p.graph_id}><strong>{p.cascade_id}: {(p.viral_probability*100).toFixed(1)}% viral probability</strong><p>Target: ≥{p.size_threshold} recorded posts · First {p.observation_minutes} minutes</p>{p.estimated && <p className="warning-text">Estimated graph — experimental prediction</p>}</div>) : <p className="warning-text">{t.result.error}</p>}</div>)}
              {!!m.data.evidence?.length && <details><summary>{m.data.evidence.length} retrieved evidence excerpts</summary>{m.data.evidence.map((e: Json) => <div className="evidence" key={e.citation}><b>{e.citation}</b><p>{e.text}</p><small>{e.provenance.page ? `${e.provenance.page}, sentence ${e.provenance.sentence}` : e.path} · {e.retrieval}</small></div>)}</details>}
            </article>)}
            {activeChat && <article className="message assistant"><div className="message-label">VERITAS · WORKING</div><pre className="stream">{live || 'Retrieving evidence and preparing local inference…'}</pre></article>}
          </div>
          <form className="composer" onSubmit={e => {e.preventDefault(); void action(send);}}><div className="row between"><span className="eyebrow">{selected.length ? `${selected.length} GRAPHS ATTACHED` : 'ASK OR ATTACH A POST DATASET'}</span><button type="button" className="link" onClick={() => void action(importPosts)}>＋ Import JSON / JSONL</button></div><textarea aria-label="Message" placeholder="Ask about a claim, a message, or its potential reach…" value={text} onChange={e => setText(e.target.value)} rows={3}/><div className="row between"><small>Evidence-based verdicts · Local model · Reviewed graph estimates</small>{activeChat ? <button type="button" onClick={() => void action(() => api('POST', `/v1/jobs/${activeChat}/cancel`, {}))}>Stop</button> : <button className="primary" disabled={!text.trim() || busy} type="submit">Send message ↑</button>}</div></form>
        </section>
        <aside className="graph-panel"><div className="row between"><h3>Post datasets</h3><span className="count">{graphs.length}</span></div><p className="muted">Select the cascades this conversation can analyze.</p><input placeholder="Filter source or cascade…" value={graphFilter} onChange={e => setGraphFilter(e.target.value)}/><div className="graph-list">{visibleGraphs.map(g => <div className="graph-card" key={g.id}><label><input type="checkbox" checked={selected.includes(g.id)} onChange={e => setSelected(e.target.checked ? [...selected, g.id] : selected.filter(id => id !== g.id))}/><strong>{g.cascade_id}</strong></label><small>{g.nodes} posts · {g.source.split(/[\\/]/).pop()}</small>{g.needs_estimation ? <span className="warning-text">Metadata incomplete</span> : g.estimated ? <span className="warning-text">Estimated graph</span> : <span className="observed">Observed graph</span>}<button className="link" onClick={() => void action(() => inspectGraph(g.id))}>Inspect / review ↗</button></div>)}</div>{!graphs.length && <div className="empty-small">Import a JSON post bundle or Twitter15/16 dataset to begin.</div>}<div className="note"><strong>Truth ≠ reach</strong><p>GraphSAGE predicts recorded cascade growth. It does not establish whether a statement is true.</p></div></aside>
      </div>}
      {view === 'Resources' && <div className="page"><section className="hero-card"><span className="eyebrow">YOUR EVIDENCE LIBRARY</span><h2>Ground every answer in a source.</h2><p>Import structured files from your resources directory. Evidence stays local, with source references attached to retrieved excerpts.</p><div className="row"><input aria-label="Resource directory" value={resourcePath} onChange={e => setResourcePath(e.target.value)}/><button onClick={() => void action(async () => {const p = await pick('directory'); if (p) setResourcePath(p);})}>Browse</button><button className="primary" onClick={() => void action(() => api('POST', '/v1/resources/index', {path: resourcePath}))}>Index directory</button></div><small>TXT · Markdown · JSON · JSONL · CSV. No PDF or OCR in this version.</small></section><div className="grid-two"><section className="card"><h3>FEVER evidence corpus</h3><p>Download historical Wikipedia evidence for supported, refuted, and insufficient-evidence judgments. Claim labels are kept outside the evidence index.</p><div className="note">Large download: compressed corpus plus extracted files and search indexes. Free space is checked during setup. Evidence reflects the historical FEVER snapshot.</div><button onClick={() => void action(() => api('POST', '/v1/resources/fever', {}))}>Download & index FEVER</button></section><section className="card"><h3>Inspect retrieval</h3><p>Search the same local evidence used by the assistant.</p><form onSubmit={e => {e.preventDefault(); void action(async () => setEvidence(await api('GET', '/v1/resources/search?q='+encodeURIComponent(query))));}}><input aria-label="Evidence query" value={query} onChange={e => setQuery(e.target.value)} placeholder="A claim or research question"/><button type="submit">Search evidence</button></form></section></div>{evidence.map(e => <div className="card" key={e.citation}><span className="citation">{e.citation} · {e.retrieval}</span><p>{e.text}</p><small>{e.path} · {json(e.provenance)}</small></div>)}</div>}
      {view === 'Skills' && <div className="page"><div className="row between"><div><h2>Give the assistant a new capability.</h2><p className="muted">Local instructions connect Qwen to registered tools.</p></div><button onClick={() => void action(async () => {const p = await pick('directory'); if (p) {await api('POST', '/v1/skills/import', {path: p}); await refresh();}})}>＋ Import skill folder</button></div>{skills.map(s => <section className="card" key={s.id}><div className="row between"><h3>◇ {s.name}</h3><label className="row"><input type="checkbox" checked={s.enabled} disabled={!!s.error} onChange={e => void action(async () => {await api('PATCH', `/v1/skills/${s.id}`, {enabled: e.target.checked}); await refresh();})}/>Enabled</label></div>{s.error ? <p className="warning-text">{s.error}</p> : <><p>{s.tools.map((t: Json) => t.handler).join(' · ')}</p><details><summary>Skill instructions</summary><pre>{s.instructions}</pre></details></>}<small>{s.path}</small></section>)}<div className="note">Skills contain SKILL.md and tools.json. They can bind to GraphSAGE prediction and evidence search; arbitrary executable plugins are not loaded.</div></div>}
      {view === 'Training' && <div className="page"><div className="row between"><div><span className="eyebrow">REPRODUCIBLE EXPERIMENTS</span><h2>Train your propagation model.</h2><p className="muted">Native Twitter15/16 import → early observation → binary virality prediction.</p></div><button onClick={() => void action(async () => {const p = await pick('directory'); if (p) await api('POST', '/v1/twitter/import', {path: p});})}>Import Twitter15 / Twitter16</button></div><div className="grid-two"><section className="card"><h3>Training configuration</h3><p>{graphs.filter(g => ['twitter15', 'twitter16'].includes(g.source)).length} native cascades imported. Rumor labels are excluded from the virality target.</p><div className="field-grid"><label>Viral at ≥ posts<input type="number" min={2} value={threshold} onChange={e => setThreshold(+e.target.value)}/></label><label>Observation (minutes)<input type="number" min={1} value={observation} onChange={e => setObservation(+e.target.value)}/></label><label>Maximum epochs<input type="number" min={1} max={2000} value={epochs} onChange={e => setEpochs(+e.target.value)}/></label><label>Random seed<input disabled value="42"/></label></div><div className="note">CPU training · GraphSAGE, 2 layers · 70/15/15 event-level split · Validation-selected cutoff. Training pauses the chat runtime to free memory.</div><button className="primary" disabled={busy} onClick={() => void action(() => api('POST', '/v1/training', {threshold, window: observation, epochs, seed: 42, device: 'cpu'}))}>Start training →</button></section><section className="card"><h3>Training progress</h3>{latestTraining ? <><span className={'pill '+(latestTraining.status === 'failed' ? 'warning-text' : '')}>{latestTraining.status}</span><LossChart events={latestTraining.events}/><p className="warning-text">{latestTraining.error}</p>{!terminal.has(latestTraining.status) && <button onClick={() => void action(() => api('POST', `/v1/jobs/${latestTraining.id}/cancel`, {}))}>Cancel training</button>}</> : <div className="empty-small">Training curves and validation loss will appear here.</div>}</section></div><h3>Saved checkpoints</h3>{!checkpoints.length && <p className="muted">No checkpoints yet. Import native datasets and start a training run.</p>}{checkpoints.map(c => <section className="card" key={c.id}><div className="row between"><div><h3>GraphSAGE · {c.id.slice(0, 8)}</h3><small>≥{c.config.threshold} posts · {c.config.window} minutes · seed {c.config.seed}</small></div><button className={status?.checkpoint === c.id ? 'selected-button' : ''} onClick={() => void action(async () => {await api('PATCH', '/v1/settings', {checkpoint: c.id}); await refresh();})}>{status?.checkpoint === c.id ? 'Selected ✓' : 'Use checkpoint'}</button></div><table><thead><tr><th>Held-out test</th><th>F1</th><th>Precision</th><th>Recall</th><th>PR-AUC</th></tr></thead><tbody>{[['GraphSAGE', c.report.test], ['Majority baseline', c.report.majority_baseline], ['Observed-size baseline', c.report.observed_size_baseline]].map(([label, m]) => <tr key={label as string}><td>{label as string}</td><td>{m.f1.toFixed(3)}</td><td>{m.precision.toFixed(3)}</td><td>{m.recall.toFixed(3)}</td><td>{m.pr_auc.toFixed(3)}</td></tr>)}</tbody></table><details><summary>Full metrics, split assignments, and provenance</summary><pre>{json(c)}</pre></details></section>)}</div>}
      {view === 'Models' && <div className="page"><h2>Your local inference stack.</h2><p className="muted">Models download automatically during initial setup. Retry interrupted downloads here or import a local GGUF.</p><div className="grid-two">{[['qwen', 'Qwen3.5-9B', 'Q4_K_M GGUF · llama.cpp'], ['embedding', 'BGE-small-en-v1.5', 'Local English evidence embeddings']].map(([key, name, description]) => <section className="card" key={key}><div className="row between"><span className="model-symbol">⬡</span><span className="pill">{status?.models[key]?.ready ? 'Downloaded' : 'Not downloaded'}</span></div><h3>{name}</h3><p>{description}</p><small>{status?.models[key]?.path}</small><button onClick={() => void action(() => api('POST', `/v1/models/${key}/download`, {}))}>Download / verify</button></section>)}</div><section className="card"><h3>Runtime settings</h3><p>GPU acceleration is preferred when the bundled llama.cpp build supports it; CPU is the fallback. A 9B model needs substantial memory and can be slow on CPU.</p><div className="row"><button onClick={() => void action(async () => {const p = await pick('gguf'); if (p) {await api('POST', '/v1/models/import', {path: p}); await refresh();}})}>Import local GGUF</button><button onClick={() => void action(async () => {const p = await pick('executable'); if (p) {await api('PATCH', '/v1/settings', {llama_path: p}); await refresh();}})}>Choose llama-server</button><button onClick={() => void action(() => api('POST', '/v1/models/smoke', {}))}>Verify generation & tools</button></div><p className="path">{status?.llama_path || 'No llama-server executable configured'}<br/>{status?.qwen_path || 'No Qwen model configured'}</p><div className="field-grid"><label>Context tokens<select value={status?.context || 8192} onChange={e => void action(async () => {await api('PATCH', '/v1/settings', {context: +e.target.value}); await refresh();})}>{[2048, 4096, 8192, 16384, 32768].map(n => <option key={n}>{n}</option>)}</select></label><label>Acceleration<select value={String(status?.gpu_layers ?? 'auto')} onChange={e => void action(async () => {await api('PATCH', '/v1/settings', {gpu_layers: e.target.value}); await refresh();})}><option value="auto">Automatic GPU offload</option><option value="0">CPU only</option></select></label></div></section></div>}
      <footer className="jobs-bar"><div className="row"><span className={'status-dot '+(activeJobs.length ? 'working' : '')}/><strong>{activeJobs.length ? `${activeJobs.length} background job${activeJobs.length > 1 ? 's' : ''}` : 'Workspace ready'}</strong><span className="muted">{activeJobs[0]?.kind || 'Local data · Local inference'}</span></div><details><summary>Activity ({jobs.length})</summary><div className="job-popover">{jobs.map(j => <div className="job-row" key={j.id}><button className="link" onClick={() => setJobDetail(j)}>{j.kind}</button><span>{j.status}</span>{!terminal.has(j.status) && <button className="link" onClick={() => void action(() => api('POST', `/v1/jobs/${j.id}/cancel`, {}))}>Cancel</button>}{j.error && <p className="warning-text">{j.error}</p>}{j.events.filter((e: Json) => e.event === 'download').slice(-1).map((e: Json) => <small key={e.seq}>{e.file} · {(e.downloaded/1024**2).toFixed(0)} MB {e.total ? `/ ${(e.total/1024**2).toFixed(0)} MB` : ''}</small>)}</div>)}</div></details></footer>
    </main>
    {importing && <Modal title="Map your post dataset" close={() => setImporting(null)}><p>{importing.count} records · {importing.path}</p><div className="field-grid">{Object.keys(importing.mapping).map(field => <label key={field}>{field}<select value={importing.mapping[field] || ''} onChange={e => setImporting({...importing, mapping: {...importing.mapping, [field]: e.target.value || null}})}><option value="">Missing</option>{Object.keys(importing.sample[0]).map(key => <option key={key}>{key}</option>)}</select></label>)}</div><label className="row"><input type="checkbox" checked={combine} onChange={e => setCombine(e.target.checked)}/>Group these posts as one cascade for graph estimation</label><details><summary>Preview source records</summary><pre>{json(importing.sample)}</pre></details><div className="row"><button disabled={busy} onClick={() => void action(async () => {const j = await api('POST', '/v1/posts/map', {path: importing.path}); const result = await waitFor(j.job_id); setImporting({...importing, mapping: result.mapping});})}>Ask Qwen to map fields</button><button className="primary" disabled={busy} onClick={() => void action(confirmImport)}>Import posts</button></div></Modal>}
    {inspecting && <Modal title={'Cascade '+inspecting.cascade_id} close={() => {setInspecting(null); setProposal(null);}}><p>{inspecting.nodes.length} posts · {inspecting.edges.length} observed/accepted links</p>{inspecting.warnings.map((w: string) => <p className="warning-text" key={w}>{w}</p>)}<details open><summary>{proposal ? 'Estimated graph proposal — review before accepting' : 'Current graph'}</summary><pre>{json(proposal || inspecting)}</pre></details>{proposal ? <><div className="alert">These links and/or timestamps are estimates. Predictions are experimental and are not validated as observed propagation forecasts.</div><button className="primary" onClick={() => void action(async () => {await api('POST', `/v1/graphs/${inspecting.id}/accept`, {accept: true}); setInspecting(null); setProposal(null); await refresh();})}>Accept this estimated graph</button></> : inspecting.needs_estimation && <button disabled={busy} onClick={() => void action(async () => {const j = await api('POST', `/v1/graphs/${inspecting.id}/estimate`, {}); setProposal(await waitFor(j.job_id));})}>Ask Qwen to propose missing metadata</button>}</Modal>}
    {jobDetail && <Modal title={jobDetail.kind+' · '+jobDetail.status} close={() => setJobDetail(null)}><pre>{json(jobDetail)}</pre></Modal>}
    {renaming !== null && <Modal title="Rename conversation" close={() => setRenaming(null)}><form onSubmit={e => {e.preventDefault(); void action(async () => {await api('PATCH', `/v1/conversations/${conversation}`, {title: renaming}); setRenaming(null); await refresh();});}}><input aria-label="Conversation name" value={renaming} onChange={e => setRenaming(e.target.value)} maxLength={200}/><button type="submit" className="primary">Save name</button></form></Modal>}
  </div>;
}

function Modal({title, close, children}: {title: string; close: () => void; children: React.ReactNode}) {
  return <div className="modal-backdrop"><section className="modal" role="dialog" aria-modal="true" aria-label={title}><div className="row between"><h2>{title}</h2><button aria-label="Close dialog" onClick={close}>×</button></div>{children}</section></div>;
}
function LossChart({events}: {events: Json[]}) {
  const history = events.filter(e => e.event === 'training');
  if (!history.length) return <p className="muted">Waiting for the training worker…</p>;
  const max = Math.max(...history.flatMap(e => [e.loss, e.validation_loss]), .01);
  const points = (key: string) => history.map((e, i) => `${20+i/Math.max(history.length-1, 1)*360},${170-e[key]/max*140}`).join(' ');
  return <><svg className="loss-chart" viewBox="0 0 400 200" role="img" aria-label="Training and validation loss"><path d="M20 20V170H380" fill="none" stroke="#35404b"/><polyline points={points('loss')} fill="none" stroke="#7fe0bd" strokeWidth="2"/><polyline points={points('validation_loss')} fill="none" stroke="#e6b674" strokeWidth="2"/></svg><small>Epoch {history.at(-1).epoch} · Training {history.at(-1).loss.toFixed(4)} · Validation {history.at(-1).validation_loss.toFixed(4)}</small></>;
}

createRoot(document.getElementById('root')!).render(<App/>);
