import { useTelemetry } from './useTelemetry';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { CheckCircleIcon, XCircleIcon } from 'lucide-react';

export default function App() {
  const { data, connected, history } = useTelemetry();

  return (
    <div className="bg-slate-900 text-slate-100 p-4">
      <h1 className="text-3xl font-bold mb-4">RLM Telemetry Dashboard</h1>
      <div className="bg-slate-800 p-4 rounded mb-4">
        <div className="flex items-center">
          {connected ? (
            <CheckCircleIcon className="text-green-500 mr-2" />
          ) : (
            <XCircleIcon className="text-red-500 mr-2" />
          )}
          <span>Status: {connected ? 'Connected (10Hz Stream)' : 'Disconnected - Retrying...'}</span>
        </div>
      </div>
      
      {data && (
        <div className="grid grid-cols-2 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-slate-800 p-4 rounded hover:scale-105"
          >
            <h3 className="text-xl font-bold mb-2">Gate Value (g_t)</h3>
            <p className="text-2xl mb-0">{data.g_t}</p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-slate-800 p-4 rounded hover:scale-105"
          >
            <h3 className="text-xl font-bold mb-2">Active Core</h3>
            <p className="text-2xl mb-0">Core {data.active_core}</p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-slate-800 p-4 rounded hover:scale-105"
          >
            <h3 className="text-xl font-bold mb-2">Free RAM</h3>
            <p className="text-2xl mb-0">{(data.memory_free_bytes / 1024).toFixed(1)} KB</p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-slate-800 p-4 rounded hover:scale-105"
          >
            <h3 className="text-xl font-bold mb-2">Window Size</h3>
            <p className="text-2xl mb-0">{data.current_window}</p>
          </motion.div>
        </div>
      )}
      {history.length > 0 && (
        <div className="mt-4">
          <LineChart width={500} height={300} data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="current_window" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="latency_ms" stroke="#8884d8" />
            <Line type="monotone" dataKey="g_t" stroke="#82ca9d" />
          </LineChart>
        </div>
      )}
    </div>
  );
}
