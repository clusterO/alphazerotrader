import { useState, useEffect, useRef } from 'react';
import { Play, Square, Settings, Terminal, TrendingUp, Trash2, Database, Box, X, Save, Activity, LayoutDashboard, ListMusic } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'logs'>('dashboard');
  const [status, setStatus] = useState({ train: 'stopped', paper: 'stopped' });
  const [market, setMarket] = useState<any>({ symbol: 'BTC/USDT', price: 0, change: 0 });
  const [logs, setLogs] = useState<string[]>(["[System] Initialize C2 Command Center...", "[System] Ready."]);
  const [config, setConfig] = useState<any>(null);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [tradeType, setTradesType] = useState<'paper' | 'backtest'>('paper');
  const [models, setModels] = useState<string[]>([]);
  const [memories, setMemories] = useState<string[]>([]);
  const [dataFiles, setDataFiles] = useState<string[]>([]);
  const [trades, setTrades] = useState<any[]>([]);

  const logEndRef = useRef<HTMLDivElement>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/status');
      const data = await res.json();
      setStatus(data);

      const resMarket = await fetch('http://localhost:8000/market');
      const dataMarket = await resMarket.json();
      if (!dataMarket.error) setMarket(dataMarket);
    } catch (e) {}
  };

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8000/models');
      setModels(await res.json());
      const resMem = await fetch('http://localhost:8000/memory');
      setMemories(await resMem.json());
      const resData = await fetch('http://localhost:8000/data_files');
      setDataFiles(await resData.json());
    } catch (e) {}
  };

  const fetchTrades = async () => {
    try {
      const res = await fetch(`http://localhost:8000/trades/${tradeType}`);
      const data = await res.json();
      setTrades(Array.isArray(data) ? data : []);
    } catch (e) {}
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch('http://localhost:8000/config');
      const data = await res.json();
      setConfig(data);
    } catch (e) {}
  };

  const saveConfig = async () => {
    await fetch('http://localhost:8000/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    setIsConfigOpen(false);
    setLogs(prev => [...prev, "[System] Configuration updated."]);
  };

  const controlProcess = async (proc: string, action: string) => {
    await fetch(`http://localhost:8000/control/${proc}/${action}`, { method: 'POST' });
    fetchStatus();
  };

  const clearLogs = async () => {
    await fetch('http://localhost:8000/logs/train', { method: 'DELETE' });
    await fetch('http://localhost:8000/logs/paper', { method: 'DELETE' });
    setLogs(["[System] Logs cleared."]);
  };

  useEffect(() => {
    fetchStatus();
    fetchModels();
    fetchTrades();
    fetchConfig();

    const fetchLogs = async () => {
      try {
        const resTrain = await fetch('http://localhost:8000/logs/tail/train');
        const logsTrain = await resTrain.json();
        const resPaper = await fetch('http://localhost:8000/logs/tail/paper');
        const logsPaper = await resPaper.json();
        setLogs([...logsTrain, ...logsPaper]);
      } catch (e) {}
    };
    fetchLogs();

    const interval = setInterval(() => {
      fetchStatus();
      fetchTrades();
    }, 3000);

    const ws = new WebSocket('ws://localhost:8000/ws/logs');
    ws.onmessage = (event) => {
      setLogs((prev) => [...prev.slice(-500), event.data]);
    };

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [tradeType]);

  useEffect(() => {
    if (activeTab === 'logs') {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, activeTab]);

  const ActionButton = ({ icon: Icon, color, onClick, label, active = false }: any) => (
    <motion.button
      whileHover={{ scale: 1.05, backgroundColor: 'rgba(0, 255, 255, 0.1)', boxShadow: '0 0 15px rgba(0, 255, 255, 0.1)' }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className={`p-2 rounded transition-all flex items-center gap-2 border border-transparent hover:border-cyan-500/30 ${active ? 'bg-cyan-500/20 text-cyan-400' : `text-${color}-500`}`}
      title={label}
    >
      <Icon size={18} />
      {label && <span className="text-[10px] font-bold uppercase tracking-widest">{label}</span>}
    </motion.button>
  );

  return (
    <div className="h-screen bg-[#020202] text-cyan-400 font-mono flex flex-col overflow-hidden relative">
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-900/10 blur-[120px] rounded-full pointer-events-none" />

      {/* Header - Fixed to top */}
      <header className="sticky top-0 z-[50] flex-shrink-0 flex justify-between items-center px-8 py-6 border-b border-cyan-900/30 bg-black/80 backdrop-blur-xl">
        <div className="flex items-center gap-8">
          <div>
            <h1 className="text-2xl font-black tracking-tighter italic text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
              AZT C2
            </h1>
            <p className="text-[8px] text-cyan-800 tracking-[0.3em] uppercase">Control Node v1.0.5</p>
          </div>
          
          <nav className="flex gap-2">
            {[
              { id: 'dashboard', label: 'DASHBOARD', icon: LayoutDashboard },
              { id: 'logs', label: 'SYSTEM LOGS', icon: Terminal }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-sm text-[10px] font-bold tracking-widest transition-all ${
                  activeTab === tab.id 
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30' 
                  : 'text-cyan-900 hover:text-cyan-600'
                }`}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 px-4 py-1.5 bg-black/60 border border-cyan-900/30 rounded-sm">
            <Activity size={14} className="text-cyan-700" />
            <div className="flex flex-col">
              <span className="text-[9px] text-cyan-800 leading-none">MARKET</span>
              <span className="text-[11px] font-bold">${market.price?.toLocaleString()}</span>
            </div>
            <span className={`text-[10px] ${market.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {market.change >= 0 ? '▲' : '▼'} {Math.abs(market.change || 0).toFixed(2)}%
            </span>
          </div>

          <div className="flex gap-2">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-sm border ${status.train === 'running' ? 'bg-green-500/5 border-green-500/20 text-green-500' : 'bg-red-500/5 border-red-500/20 text-red-500'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${status.train === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-[9px] font-bold">TRAIN</span>
            </div>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-sm border ${status.paper === 'running' ? 'bg-green-500/5 border-green-500/20 text-green-500' : 'bg-red-500/5 border-red-500/20 text-red-500'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${status.paper === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-[9px] font-bold">PAPER</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 min-h-0 relative flex gap-6 p-8 w-full overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' ? (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex gap-6 h-full w-full min-h-0"
            >
              {/* Left Sidebar: Controls & Models */}
              <div className="w-[350px] flex-shrink-0 space-y-6 flex flex-col h-full overflow-hidden">
                <div className="bg-black/40 backdrop-blur-md border border-cyan-900/20 p-6 rounded-sm shadow-2xl">
                  <h2 className="text-[10px] font-black mb-6 text-cyan-700 tracking-[0.2em] flex items-center gap-2 uppercase">
                    <Database size={14}/> Execution_Core
                  </h2>
                  <div className="space-y-6">
                    <div className="p-4 bg-white/[0.02] border border-white/[0.05] rounded-sm">
                      <p className="text-[9px] text-cyan-900 mb-3 uppercase tracking-widest font-bold">Training Cluster</p>
                      <div className="flex gap-4">
                        <ActionButton icon={Play} color="green" onClick={() => controlProcess('train', 'start')} label="START" />
                        <ActionButton icon={Square} color="red" onClick={() => controlProcess('train', 'stop')} label="STOP" />
                      </div>
                    </div>
                    <div className="p-4 bg-white/[0.02] border border-white/[0.05] rounded-sm">
                      <p className="text-[9px] text-cyan-900 mb-3 uppercase tracking-widest font-bold">Paper Node</p>
                      <div className="flex gap-4">
                        <ActionButton icon={Play} color="green" onClick={() => controlProcess('paper', 'start')} label="START" />
                        <ActionButton icon={Square} color="red" onClick={() => controlProcess('paper', 'stop')} label="STOP" />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-black/40 backdrop-blur-md border border-cyan-900/20 p-6 rounded-sm shadow-2xl flex-1 flex flex-col min-h-0 overflow-hidden">
                  <h2 className="text-[10px] font-black mb-4 text-blue-700 tracking-[0.2em] flex items-center gap-2 uppercase">
                    <Box size={14}/> Registry_System
                  </h2>
                  <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar text-[10px] space-y-8">
                    {/* Models */}
                    <div className="space-y-2">
                      <p className="text-[8px] text-cyan-900 uppercase font-black tracking-widest border-l border-cyan-900 pl-2">Neural_Models</p>
                      <div className="max-h-[160px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {models.map((m, idx) => (
                          <div key={m} className="flex items-center gap-3 p-3 bg-blue-500/5 border border-blue-500/10 rounded-sm group hover:border-blue-500/40 transition-all cursor-crosshair">
                            <Database size={12} className="text-blue-900 group-hover:text-blue-500 flex-shrink-0" />
                            <span className="truncate flex-1 text-blue-300/70 group-hover:text-blue-300">{m}</span>
                            {idx === 0 && <span className="text-[7px] bg-blue-500/20 px-1.5 py-0.5 rounded-full text-blue-400 border border-blue-500/30 font-black flex-shrink-0">ACTIVE</span>}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Memories */}
                    <div className="space-y-2">
                      <p className="text-[8px] text-cyan-900 uppercase font-black tracking-widest border-l border-cyan-900 pl-2">Memory_Vault</p>
                      <div className="max-h-[140px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {memories.map(m => (
                          <div key={m} className="flex items-center gap-3 p-2 bg-purple-500/5 border border-purple-500/10 rounded-sm group hover:border-purple-500/40 transition-all">
                            <Activity size={12} className="text-purple-900 group-hover:text-purple-500 flex-shrink-0" />
                            <span className="truncate flex-1 text-purple-300/70 group-hover:text-purple-300">{m}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Data Files */}
                    <div className="space-y-2">
                      <p className="text-[8px] text-cyan-900 uppercase font-black tracking-widest border-l border-cyan-900 pl-2">Data_Buffer</p>
                      <div className="max-h-[140px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {dataFiles.map(f => (
                          <div key={f} className="flex items-center gap-3 p-2 bg-green-500/5 border border-green-500/10 rounded-sm group hover:border-green-500/40 transition-all">
                            <ListMusic size={12} className="text-green-900 group-hover:text-green-500 flex-shrink-0" />
                            <span className="truncate flex-1 text-green-300/70 group-hover:text-green-300">{f}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <motion.button 
                    whileHover={{ scale: 1.02, backgroundColor: 'rgba(245, 158, 11, 0.1)' }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setIsConfigOpen(true)} 
                    className="w-full mt-6 py-3 border border-amber-500/20 text-amber-500/60 text-[10px] hover:text-amber-500 transition-colors uppercase font-black tracking-[0.2em] flex items-center justify-center gap-2 rounded-sm flex-shrink-0"
                  >
                    <Settings size={14}/> OPEN CONFIG_SYSTEM
                  </motion.button>
                </div>
              </div>

              {/* Main Workspace: Charts & History */}
              <div className="flex-1 flex flex-col gap-6 min-w-0 h-full">
                <div className="bg-black/40 backdrop-blur-md border border-cyan-900/20 p-6 rounded-sm shadow-2xl h-[300px] flex-shrink-0 relative overflow-hidden">
                  <h2 className="text-[10px] font-black mb-4 text-blue-500 tracking-[0.2em] uppercase flex items-center gap-2">
                    <TrendingUp size={14}/> Visual_Telemetry
                  </h2>
                  <div className="absolute inset-0 flex items-center justify-center opacity-[0.03] pointer-events-none">
                    <TrendingUp size={240} />
                  </div>
                  <div className="h-full w-full flex items-end pb-8">
                    <svg className="w-full h-48 text-cyan-500/20" viewBox="0 0 1000 100" preserveAspectRatio="none">
                      <motion.path 
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 2 }}
                        d="M0,80 L50,85 L100,70 L150,75 L200,50 L250,60 L300,40 L350,45 L400,20 L450,30 L500,10 L550,15 L600,25 L650,5 L700,12 L750,8 L800,20 L850,5 L900,15 L1000,2" 
                        fill="none" 
                        stroke="currentColor" 
                        strokeWidth="1" 
                      />
                    </svg>
                  </div>
                </div>

                <div className="bg-black/40 backdrop-blur-md border border-cyan-900/20 p-6 rounded-sm shadow-2xl flex-1 flex flex-col min-h-0 overflow-hidden">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-[10px] font-black text-green-500 tracking-[0.2em] uppercase flex items-center gap-2">
                      <ListMusic size={14}/> Trade_Intelligence_Journal
                    </h2>
                    <div className="flex gap-1 bg-black/60 p-1 border border-white/5 rounded-sm">
                      <button onClick={() => setTradesType('paper')} className={`text-[9px] px-4 py-1.5 transition-all font-black tracking-widest ${tradeType === 'paper' ? 'bg-cyan-500/20 text-cyan-400' : 'text-cyan-950 hover:text-cyan-800'}`}>LIVE_PAPER</button>
                      <button onClick={() => setTradesType('backtest')} className={`text-[9px] px-4 py-1.5 transition-all font-black tracking-widest ${tradeType === 'backtest' ? 'bg-cyan-500/20 text-cyan-400' : 'text-cyan-950 hover:text-cyan-800'}`}>BACKTEST_ARCHIVE</button>
                    </div>
                  </div>
                  <div className="flex-1 overflow-auto pr-4 custom-scrollbar">
                    <table className="w-full text-left min-w-full">
                      <thead className="sticky top-0 bg-[#0a0a0a] z-10 border-b border-cyan-900/30">
                        <tr className="text-[9px] text-cyan-900 font-black uppercase tracking-widest">
                          <th className="py-3 px-2">Temporal_Stamp</th>
                          <th className="py-3 px-2">Decision</th>
                          <th className="py-3 px-2 text-center">Confidence</th>
                          <th className="py-3 px-2 text-right">Balance</th>
                        </tr>
                      </thead>
                      <tbody className="text-[10px]">
                        {trades.slice().reverse().map((t, i) => (
                          <motion.tr 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            key={i} 
                            className="border-b border-white/5 hover:bg-white/[0.02] transition-colors group"
                          >
                            <td className="py-3 px-2 text-cyan-950 group-hover:text-cyan-800 transition-colors font-mono whitespace-nowrap">{t.timestamp || t.date || '---'}</td>
                            <td className={`py-3 px-2 font-black whitespace-nowrap ${t.action == 1 ? 'text-green-500' : t.action == 2 ? 'text-red-500' : 'text-cyan-950'}`}>
                              {t.action == 1 ? '>> LONG' : t.action == 2 ? '<< SHORT' : '-- FLAT'}
                            </td>
                            <td className="py-3 px-2 text-center text-cyan-700/50">{parseFloat(t.mcts_v || 0).toFixed(6)}</td>
                            <td className="py-3 px-2 text-right font-black text-cyan-400/80 group-hover:text-cyan-400 whitespace-nowrap">${parseFloat(t.balance || 0).toLocaleString()}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="logs"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full w-full flex flex-col bg-black/40 backdrop-blur-md border border-cyan-900/20 rounded-sm overflow-hidden"
            >
              <div className="p-4 border-b border-white/5 flex items-center justify-between bg-black/20">
                <div className="flex items-center gap-4">
                  <h2 className="text-xs font-black tracking-[0.2em] flex items-center gap-2 uppercase">
                    <Terminal size={16} className="text-cyan-500" /> Unified_System_Output
                  </h2>
                  <span className="text-[8px] bg-green-500/10 text-green-500 px-2 py-0.5 border border-green-500/20 rounded-full animate-pulse">STREAM_ACTIVE</span>
                </div>
                <div className="flex gap-4">
                  <button onClick={clearLogs} className="flex items-center gap-2 px-4 py-1.5 rounded-sm border border-red-500/30 text-red-500/50 hover:text-red-500 hover:bg-red-500/10 text-[9px] font-black transition-all">
                    <Trash2 size={12}/> PURGE_BUFFER
                  </button>
                </div>
              </div>
              <div className="flex-1 p-6 text-[10px] overflow-y-auto space-y-1.5 font-mono custom-scrollbar leading-relaxed">
                {logs.map((log, i) => (
                  <div key={i} className={`flex gap-4 ${log.includes('[PAPER]') ? 'text-cyan-400/80' : log.includes('[TRAIN]') ? 'text-blue-400/80' : 'text-cyan-900'}`}>
                    <span className="opacity-30 flex-shrink-0">{(i+1).toString().padStart(4, '0')}</span>
                    <span className="break-all">{log}</span>
                  </div>
                ))}
                <div ref={logEndRef} />
                <div className="animate-pulse w-2 h-4 bg-cyan-500/50 inline-block mt-2" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Config Modal */}
      {isConfigOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-xl p-8">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-[#050505] border border-cyan-500/20 w-full max-w-5xl max-h-[85vh] flex flex-col rounded-sm shadow-[0_0_100px_rgba(0,255,255,0.05)]"
          >
            <div className="p-8 border-b border-white/5 flex justify-between items-center bg-black/40">
              <div>
                <h2 className="text-xl font-black text-cyan-400 uppercase tracking-[0.3em] flex items-center gap-3">
                  <Settings className="animate-spin-slow text-amber-500" size={24} /> System_Manifest
                </h2>
                <p className="text-[9px] text-cyan-900 mt-1 uppercase tracking-widest">Override Kernel Parameters</p>
              </div>
              <button onClick={() => setIsConfigOpen(false)} className="p-2 hover:bg-red-500/10 text-cyan-900 hover:text-red-500 rounded-full transition-all"><X size={24}/></button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-10 grid grid-cols-3 gap-10 custom-scrollbar">
              {config && Object.entries(config).map(([section, params]: [string, any]) => (
                <div key={section} className="space-y-6">
                  <h3 className="text-[10px] font-black text-amber-500 border-b border-amber-500/20 pb-2 uppercase tracking-[0.2em]">{section}_PROTOCOL</h3>
                  <div className="space-y-5">
                    {Object.entries(params).map(([key, val]: [string, any]) => (
                      <div key={key} className="flex flex-col gap-2 group">
                        <label className="text-[9px] text-cyan-950 uppercase font-black group-hover:text-cyan-700 transition-colors">{key}</label>
                        <input 
                          type={typeof val === 'number' ? 'number' : 'text'}
                          value={Array.isArray(val) ? JSON.stringify(val) : val}
                          onChange={(e) => {
                            const newConfig = { ...config };
                            let newVal: any = e.target.value;
                            if (typeof val === 'number') newVal = parseFloat(newVal);
                            // Simple array handling for range
                            if (Array.isArray(val)) {
                                try { newVal = JSON.parse(newVal); } catch(e) {}
                            }
                            newConfig[section][key] = newVal;
                            setConfig(newConfig);
                          }}
                          className="bg-black/60 border border-white/5 p-3 text-[11px] text-white focus:border-cyan-500/50 focus:bg-cyan-500/5 outline-none rounded-sm transition-all font-mono"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="p-8 border-t border-white/5 flex justify-end gap-6 bg-black/40">
              <button onClick={() => setIsConfigOpen(false)} className="px-8 py-3 text-[10px] font-black uppercase text-cyan-950 hover:text-cyan-400 tracking-widest transition-colors">Discard_Changes</button>
              <motion.button 
                whileHover={{ scale: 1.05, boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)' }}
                whileTap={{ scale: 0.95 }}
                onClick={saveConfig} 
                className="bg-cyan-500/10 border border-cyan-500/40 px-12 py-3 text-[10px] uppercase font-black text-cyan-400 hover:bg-cyan-500/20 transition-all flex items-center gap-3 rounded-sm shadow-lg tracking-[0.2em]"
              >
                <Save size={16}/> Deploy_Overwrites
              </motion.button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Global CSS for scrollbars */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 3px; height: 3px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(0, 255, 255, 0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 255, 0.2); }
        .animate-spin-slow { animation: spin 12s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
          -webkit-appearance: none; 
          margin: 0; 
        }
      `}</style>
    </div>
  );
}

export default App;
