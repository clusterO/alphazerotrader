"use client";

import React, { useState, useEffect } from 'react';
import { Activity, Play, Square, Settings, Database, Terminal, TrendingUp } from 'lucide-react';

export default function Dashboard() {
  const [status, setStatus] = useState({ train: 'stopped', paper: 'stopped' });
  const [config, setConfig] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>(["[System] Initialize C2 Command Center...", "[System] Ready."]);

  // Mock fetching status
  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/status');
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error("Backend offline");
    }
  };

  const controlProcess = async (proc: string, action: string) => {
    await fetch(`http://localhost:8000/control/${proc}/${action}`, { method: 'POST' });
    fetchStatus();
  };

  useEffect(() => {
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-cyan-400 font-mono p-8 overflow-hidden relative">
      {/* Background Glows */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/20 blur-[120px] rounded-full" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-amber-900/10 blur-[120px] rounded-full" />

      {/* Header */}
      <header className="flex justify-between items-center mb-12 border-b border-cyan-900/30 pb-6 relative z-10">
        <div>
          <h1 className="text-4xl font-black tracking-tighter italic text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
            AZT COMMAND CENTER
          </h1>
          <p className="text-xs text-cyan-700 tracking-widest mt-1 uppercase">AlphaZero Trading Node v1.0.4</p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-2 bg-black/40 border border-cyan-900/50 px-4 py-2 rounded-sm">
            <div className={`w-2 h-2 rounded-full ${status.train === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-xs uppercase">Training: {status.train}</span>
          </div>
          <div className="flex items-center gap-2 bg-black/40 border border-cyan-900/50 px-4 py-2 rounded-sm">
            <div className={`w-2 h-2 rounded-full ${status.paper === 'running' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-xs uppercase">Paper: {status.paper}</span>
          </div>
        </div>
      </header>

      {/* Grid Layout */}
      <main className="grid grid-cols-12 gap-6 relative z-10">
        
        {/* Left Column: Controls & Config */}
        <section className="col-span-3 space-y-6">
          <div className="bg-black/60 backdrop-blur-xl border border-cyan-900/20 p-6 rounded-lg shadow-[0_0_20px_rgba(0,255,255,0.05)]">
            <h2 className="text-sm font-bold mb-4 flex items-center gap-2 border-l-2 border-cyan-500 pl-2">PROCESS CONTROL</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs text-cyan-600">TRAINING ENGINE</span>
                <div className="flex gap-2">
                  <button onClick={() => controlProcess('train', 'start')} className="p-2 hover:bg-cyan-500/10 rounded transition-colors text-green-500"><Play size={16}/></button>
                  <button onClick={() => controlProcess('train', 'stop')} className="p-2 hover:bg-red-500/10 rounded transition-colors text-red-500"><Square size={16}/></button>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-cyan-600">PAPER TRADER</span>
                <div className="flex gap-2">
                  <button onClick={() => controlProcess('paper', 'start')} className="p-2 hover:bg-cyan-500/10 rounded transition-colors text-green-500"><Play size={16}/></button>
                  <button onClick={() => controlProcess('paper', 'stop')} className="p-2 hover:bg-red-500/10 rounded transition-colors text-red-500"><Square size={16}/></button>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-black/60 backdrop-blur-xl border border-cyan-900/20 p-6 rounded-lg">
            <h2 className="text-sm font-bold mb-4 flex items-center gap-2 border-l-2 border-amber-500 pl-2 text-amber-500">SYSTEM PARAMETERS</h2>
            <div className="space-y-3 opacity-60">
              {['MCTS_SIMS: 100', 'LR: 0.0001', 'TAU_DECAY: 0.99', 'BATCH: 256'].map((p) => (
                <div key={p} className="text-[10px] flex justify-between border-b border-white/5 pb-1">
                  <span>{p.split(':')[0]}</span>
                  <span className="text-white">{p.split(':')[1]}</span>
                </div>
              ))}
            </div>
            <button className="w-full mt-4 py-2 border border-amber-500/30 text-amber-500 text-[10px] hover:bg-amber-500/10 transition-colors uppercase flex items-center justify-center gap-2">
              <Settings size={12}/> Configure
            </button>
          </div>
        </section>

        {/* Center: Live Chart & Trades */}
        <section className="col-span-6 space-y-6">
          <div className="bg-black/60 backdrop-blur-xl border border-cyan-900/20 p-6 rounded-lg h-[400px] flex flex-col">
            <h2 className="text-sm font-bold mb-4 flex items-center gap-2 border-l-2 border-blue-500 pl-2 text-blue-500">REAL-TIME EQUITY CURVE</h2>
            <div className="flex-1 border border-white/5 bg-gradient-to-b from-blue-500/5 to-transparent relative overflow-hidden">
               {/* Mock Chart Area */}
               <div className="absolute inset-0 flex items-center justify-center opacity-20">
                  <TrendingUp size={120} strokeWidth={0.5} />
               </div>
            </div>
          </div>

          <div className="bg-black/60 backdrop-blur-xl border border-cyan-900/20 p-6 rounded-lg">
            <h2 className="text-sm font-bold mb-4 flex items-center gap-2 border-l-2 border-green-500 pl-2 text-green-500">ACTIVE POSITIONS</h2>
            <table className="w-full text-[10px]">
              <thead>
                <tr className="text-cyan-800 text-left uppercase">
                  <th className="pb-2">Asset</th>
                  <th className="pb-2">Type</th>
                  <th className="pb-2">Entry</th>
                  <th className="pb-2">Size</th>
                  <th className="pb-2 text-right">PnL</th>
                </tr>
              </thead>
              <tbody className="text-white">
                <tr>
                  <td className="py-2">BTC/USDT</td>
                  <td className="text-green-500">LONG</td>
                  <td>64,231.20</td>
                  <td>0.421</td>
                  <td className="text-right text-green-400">+2.41%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Right Column: Terminal & Logs */}
        <section className="col-span-3 space-y-6">
          <div className="bg-black/60 backdrop-blur-xl border border-cyan-900/20 rounded-lg h-full flex flex-col min-h-[600px]">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <h2 className="text-xs font-bold flex items-center gap-2"><Terminal size={14}/> SYSTEM_LOGS</h2>
              <span className="text-[8px] bg-cyan-500/20 px-2 py-0.5 rounded-full">LIVE</span>
            </div>
            <div className="flex-1 p-4 text-[9px] overflow-y-auto space-y-1 font-mono">
              {logs.map((log, i) => (
                <div key={i} className="text-cyan-700/80">
                  <span className="text-cyan-900 mr-2">[{new Date().toLocaleTimeString()}]</span>
                  {log}
                </div>
              ))}
              <div className="animate-pulse w-1 h-3 bg-cyan-500 inline-block mt-1" />
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
