import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Mon', latency: 45, bandwidth: 78, uptime: 99.9 },
  { name: 'Tue', latency: 48, bandwidth: 82, uptime: 99.8 },
  { name: 'Wed', latency: 42, bandwidth: 75, uptime: 99.95 },
  { name: 'Thu', latency: 50, bandwidth: 85, uptime: 99.7 },
  { name: 'Fri', latency: 44, bandwidth: 80, uptime: 99.9 },
  { name: 'Sat', latency: 46, bandwidth: 88, uptime: 99.85 },
  { name: 'Sun', latency: 41, bandwidth: 72, uptime: 99.95 }
];

export const NetworkChart: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-lg font-bold text-neutralDark mb-4">
        Network Performance (Weekly)
      </h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0077B6" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#0077B6" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorBandwidth" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00B4D8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#00B4D8" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorUptime" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#FFD166" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#FFD166" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #ddd' }} />
          <Legend />
          <Area type="monotone" dataKey="latency" stroke="#0077B6" fillOpacity={1} fill="url(#colorLatency)" />
          <Area type="monotone" dataKey="bandwidth" stroke="#00B4D8" fillOpacity={1} fill="url(#colorBandwidth)" />
          <Area type="monotone" dataKey="uptime" stroke="#FFD166" fillOpacity={1} fill="url(#colorUptime)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
