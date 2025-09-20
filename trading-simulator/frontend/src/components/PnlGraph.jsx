import React from 'react';
import PropTypes from 'prop-types';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Label,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
  Brush
} from 'recharts';
import { format, parseISO } from 'date-fns';

const COLORS = ['#34d399', '#f87171']; // Green, Red

const PnlGraph = ({ data, isLoading, error, timeframe = 12 }) => {
  const chartHeight = 400;
  const aspectRatio = 2;

  if (isLoading) {
    return (
      <div className="p-4 bg-white rounded-xl shadow-md flex items-center justify-center h-[700px]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-8 w-8 bg-emerald-200 rounded-full mb-2"></div>
          <div className="text-gray-500 text-base font-medium">
            Loading financial data...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-white rounded-xl shadow-md flex items-center justify-center h-[700px]">
        <div className="text-center">
          <div className="text-red-500 font-semibold mb-2 text-lg">
            ⚠️ Data Loading Failed
          </div>
          <div className="text-sm text-gray-600 max-w-xs">{error.message}</div>
        </div>
      </div>
    );
  }

  if (!Array.isArray(data) || data.length === 0) {
    return (
      <div className="p-4 bg-white rounded-xl shadow-md flex flex-col items-center justify-center h-[700px]">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-12 w-12 text-gray-400 mb-2"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1}
            d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <div className="text-gray-600 text-base">No performance data available</div>
        <div className="text-sm text-gray-400 mt-1">
          Try adjusting your time range
        </div>
      </div>
    );
  }

  let processedData = [];
  try {
    processedData = data
      .map((item) => ({
        ...item,
        date: parseISO(item.datetime),
        value: Number(item.pnl_sum) || 0
      }))
      .sort((a, b) => a.date - b.date);
  } catch (e) {
    console.error('Data processing error:', e);
    return (
      <div className="p-4 bg-white rounded-xl shadow-md flex items-center justify-center h-[700px]">
        <div className="text-red-500">Data format error</div>
      </div>
    );
  }

  const firstValue = processedData[0]?.value || 0;
  const wins = data.filter((item) => item.pnl > 0).length;
  const losses = data.filter((item) => item.pnl < 0).length;
  const total = wins + losses;

  const pieData = [
    { name: 'Wins', value: parseFloat(((wins / total) * 100).toFixed(1)) },
    { name: 'Losses', value: parseFloat(((losses / total) * 100).toFixed(1)) }
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg border border-gray-100 text-gray-800 antialiased">
      <div className="mb-6">
         <h2 className="text-xl font-bold"> Cummulative PNL</h2>
      </div>

      {/* First Row - AreaChart & PieChart side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Area Chart with Slider */}
        <div className="bg-white p-4 rounded-xl shadow-md border border-gray-100">
          <ResponsiveContainer width="120%" height={chartHeight} aspect={aspectRatio}>
            <AreaChart
              data={processedData}
              margin={{ top: 20, right: 20, left: 20, bottom: 40 }}
            >
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={(date) => format(date, 'MMM yyyy')}
                tick={{ fill: '#4b5563', fontSize: 12 }}
                axisLine={false}
                tickMargin={10}
              />
              <YAxis
                tickFormatter={(value) => value.toLocaleString()}
                tick={{ fill: '#4b5563', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                width={70}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  fontSize: '13px'
                }}
                labelFormatter={(date) => format(date, 'dd MMM yyyy')}
                formatter={(value) => [
                  new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD'
                  }).format(value),
                  'PnL'
                ]}
              />
              <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#059669"
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#colorValue)"
              />
              <ReferenceLine y={firstValue} stroke="#cbd5e1" strokeDasharray="3 3">
                <Label
                  value="Start"
                  position="insideBottomRight"
                  fill="#6b7280"
                  fontSize={12}
                />
              </ReferenceLine>
              <Brush dataKey="date" height={25} stroke="#10b981" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Chart with Percentage */}
        <div className="bg-white p-4 rounded-xl shadow-md border border-gray-100 flex items-center justify-center">
          <ResponsiveContainer width="80%" height={chartHeight}>
            <PieChart>
              <Pie
                dataKey="value"
                isAnimationActive={true}
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Third Graph - LineChart with Slider */}
      <div className="mt-10 bg-white p-4 rounded-xl shadow-md border border-gray-100">
        <h2 className="text-xl font-bold"> Individual Trade PNL</h2>
        <ResponsiveContainer width="100%" height={500}>
          <LineChart
            data={processedData.map((item) => ({
              date: item.date,
              pnl: Number(item.pnl) || 0
            }))}
            margin={{ top: 20, right: 30, left: 30, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              tickFormatter={(date) => format(date, 'MMM yyyy')}
              tick={{ fill: '#4b5563', fontSize: 12 }}
              axisLine={false}
              tickMargin={10}
            />
            <YAxis
              tickFormatter={(value) => value.toLocaleString()}
              tick={{ fill: '#4b5563', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              width={70}
            />
            <Tooltip
              contentStyle={{
                background: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '0.5rem',
                fontSize: '13px'
              }}
              labelFormatter={(date) => format(date, 'dd MMM yyyy')}
              formatter={(value) => [
                new Intl.NumberFormat('en-US', {
                  style: 'currency',
                  currency: 'USD'
                }).format(value),
                'PnL'
              ]}
            />
            <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="pnl"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 4, stroke: '#1d4ed8', strokeWidth: 1, fill: '#fff' }}
              activeDot={{ r: 6, stroke: '#1d4ed8', strokeWidth: 2, fill: '#fff' }}
            />
            <Brush dataKey="date" height={25} stroke="#3b82f6" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex justify-end text-sm text-gray-500">
        As of {format(new Date(), 'dd MMM yyyy')}
      </div>
    </div>
  );
};

PnlGraph.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      datetime: PropTypes.string.isRequired,
      pnl_sum: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
        .isRequired
    })
  ),
  isLoading: PropTypes.bool,
  error: PropTypes.shape({
    message: PropTypes.string
  }),
  timeframe: PropTypes.number
};

PnlGraph.defaultProps = {
  data: [],
  isLoading: false,
  error: null,
  timeframe: 12
};

export default PnlGraph;
