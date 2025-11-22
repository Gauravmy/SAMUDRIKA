import React, { useState } from 'react';
import { Activity, AlertCircle, CheckCircle } from 'lucide-react';

interface EdgeDevice {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'processing';
  lastSeen: string;
  processedImages: number;
}

export const EdgePanel: React.FC = () => {
  const [devices] = useState<EdgeDevice[]>([
    {
      id: '1',
      name: 'Edge Device 01',
      status: 'online',
      lastSeen: '2 minutes ago',
      processedImages: 156
    },
    {
      id: '2',
      name: 'Edge Device 02',
      status: 'processing',
      lastSeen: 'Just now',
      processedImages: 89
    },
    {
      id: '3',
      name: 'Edge Device 03',
      status: 'offline',
      lastSeen: '1 hour ago',
      processedImages: 234
    }
  ]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'processing':
        return <Activity className="w-5 h-5 text-blue-500 animate-pulse" />;
      case 'offline':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="bg-gradient-to-r from-primary to-secondary px-6 py-4">
        <h2 className="text-lg font-bold text-white">Edge Devices</h2>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-neutral-50 border-b">
            <tr>
              <th className="px-6 py-3 text-left text-sm font-semibold text-neutralDark">
                Device Name
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold text-neutralDark">
                Status
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold text-neutralDark">
                Last Seen
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold text-neutralDark">
                Images Processed
              </th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {devices.map((device) => (
              <tr key={device.id} className="hover:bg-neutral-50 transition">
                <td className="px-6 py-4 text-sm text-neutralDark font-medium">
                  {device.name}
                </td>
                <td className="px-6 py-4 text-sm">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(device.status)}
                    <span className="capitalize text-neutralDark">
                      {device.status}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-neutral-600">
                  {device.lastSeen}
                </td>
                <td className="px-6 py-4 text-sm font-semibold text-primary">
                  {device.processedImages}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
