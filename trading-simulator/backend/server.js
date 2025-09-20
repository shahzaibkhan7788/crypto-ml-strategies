const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');

const app = express();
const port = 3001;

//things that you need to give colors to the individual pnl.so taht negative and positive base
//one more thing add filters..
// add the other things as well like time frame and strategy details as well..
//change sentences as well

const bcrypt = require("bcrypt");



// Middleware
app.use(cors());
app.use(express.json());

// PostgreSQL connection
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'exchange',
  password: 'shah7788',
  port: 5432,
});

// Test DB connection
pool.query('SELECT NOW()')
  .then(() => console.log('✅ Connected to PostgreSQL database'))
  .catch(err => console.error('❌ Database connection error:', err));

// Root endpoint
app.get('/', (req, res) => {
  res.send(`
    <h1>Trading Simulator API</h1>
    <p>Available endpoints:</p>
    <ul>
      <li><a href="/api/strategies">GET /api/strategies</a> - Get strategy summary</li>
    </ul>
  `);
});

// Combined strategies endpoint
app.get('/api/strategies', async (req, res) => {
  try {
    console.log('Fetching data from strategies_indicators...');

    // Step 1: Fetch base strategy info from public schema
    const baseQuery = `
      SELECT strategy_name, exchange, symbol, time_horizon 
      FROM public.strategies_indicators
    `;
    const baseResult = await pool.query(baseQuery);
    const baseStrategies = baseResult.rows;

    if (baseStrategies.length === 0) {
      return res.status(404).json({ success: false, error: 'No strategies found in public.strategies_indicators' });
    }

    console.log(`➡️ Found ${baseStrategies.length} strategies. Fetching associated ledger data...`);

    // Step 2: For each strategy, fetch corresponding latest row from ledger table
    const results = [];

    for (const strategy of baseStrategies) {
      const strategyKey = strategy.strategy_name; // like "strategy_00"
      const tableName = `backtest_${strategyKey}`; // matches: backtest_strategy_00

      try {
        const query = `
          SELECT datetime,pnl_sum 
          FROM ledger.${tableName} 
          ORDER BY datetime DESC 
          LIMIT 1
        `;
        const result = await pool.query(query);

        const lastRow = result.rows[0] || {};
        results.push({
          strategy_name: strategyKey,
          exchange: strategy.exchange,
          symbol: strategy.symbol,
          time_horizon: strategy.time_horizon,
          datetime: lastRow.datetime || null,
          pnl_sum: lastRow.pnl_sum || null
        });
      } catch (err) {
        console.warn(`⚠️ Failed to fetch from table ${tableName}:`, err.message);
        results.push({
          strategy_name: strategyKey,
          exchange: strategy.exchange,
          symbol: strategy.symbol,
          time_horizon: strategy.time_horizon,
          datetime: null,
          pnl_sum: null,
          error: `Could not fetch data from ${tableName}`
        });
      }
    }

    res.json({
      success: true,
      count: results.length,
      data: results
    });

  } 
  catch (error) {
    console.error('❌ Error in /api/strategies:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      details: error.message
    });
  }
});





// for backtest to get full data
// Fetch full backtest data for a given strategy

app.get("/api/backtest/:strategyName", async (req, res) => {
  const { strategyName } = req.params;

  try {
    // Step 1: Get strategy details from public schema
    const strategyQuery = `
      SELECT strategy_name, exchange, symbol, time_horizon
      FROM public.strategies_indicators
      WHERE strategy_name = $1
    `;
    const strategyResult = await pool.query(strategyQuery, [strategyName]);

    if (strategyResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: `Strategy ${strategyName} not found`,
      });
    }

    const strategy = strategyResult.rows[0];
    const tableName = `backtest_${strategyName}`;


    // Step 2: Fetch all backtest rows from ledger schema
    const backtestQuery = `
      SELECT datetime, predicted_direction, action, buy_price, sell_price, balance, pnl, pnl_sum
      FROM ledger.${tableName}
      ORDER BY datetime DESC
    `;
    const backtestResult = await pool.query(backtestQuery);

    res.json({
      success: true,
      strategy: {
        name: strategy.strategy_name,
        exchange: strategy.exchange,
        symbol: strategy.symbol,
        time_horizon: strategy.time_horizon,
      },
      records: backtestResult.rows,
    });
  } catch (error) {
    console.error(`❌ Error fetching backtest data for ${req.params.strategyName}:`, error);
    res.status(500).json({
      success: false,
      error: "Internal server error",
      details: error.message,
    });
  }
});




app.get('/api/strategy/:strategy_name/data', async (req, res) => {
  const { strategy_name } = req.params;
  console.log(`📥 Request received for strategy: ${strategy_name}`);

  try {
    const result = await pool.query(
      `SELECT datetime, pnl, pnl_sum FROM ledger.backtest_${strategy_name} ORDER BY datetime ASC`
    );
    console.log("📤 Sending data:", result.rows.length, "rows");

    // TEMP: Add a debug field to verify in frontend
    res.json({
      debug: `You requested ${strategy_name}`,
      data: result.rows
    });
  } catch (err) {
    console.error("❌ DB Error:", err.message);
    res.status(500).json({
      error: 'Server Error',
      message: err.message
    });
  }
});


//this is for thes status metrics 
app.get('/api/strategy/:strategy_name/full-data', async (req, res) => {
  const { strategy_name } = req.params;
  console.log(`📥 Full data request for strategy: ${strategy_name}`);

  const searchName = `backtest_${strategy_name}`;

  try {
    const result = await pool.query(`
      SELECT *
      FROM ledger_status.strategy_metrics 
      WHERE strategy_name = $1
    `, [searchName]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Strategy not found.' });
    }

    res.json({
      debug: `Returned row for ${searchName}`,
      data: result.rows[0]
    });

  } catch (err) {
    console.error("❌ DB Error:", err.message);
    res.status(500).json({
      error: 'Server Error',
      message: err.message
    });
  }
});


//this is for indicators.
app.get('/api/strategy/:strategy_name/indicators', async (req, res) => {
  const { strategy_name } = req.params;
  console.log(`📥 Indicators request for strategy: ${strategy_name}`);

  try {
    const query = `
      SELECT *
      FROM public.strategies_indicators
      WHERE strategy_name = $1
      LIMIT 1
    `;

    const result = await pool.query(query, [strategy_name]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Strategy not found in indicators table.' });
    }

    const fullRow = result.rows[0];
    const selectedData = {};

    // Always include these columns
    const alwaysInclude = ['strategy_name', 'exchange', 'symbol', 'time_horizon'];
    for (const key of alwaysInclude) {
      selectedData[key] = fullRow[key];
    }

    // Dynamically include boolean columns that are true and their corresponding `_` columns
    for (const [key, value] of Object.entries(fullRow)) {
      if (typeof value === 'boolean' && value === true) {
        selectedData[key] = true;

        // Search for keys that start with this boolean key + "_"
        const prefix = key + '_';
        for (const subKey of Object.keys(fullRow)) {
          if (subKey.startsWith(prefix)) {
            selectedData[subKey] = fullRow[subKey];
          }
        }
      }
    }

    res.json({
      debug: `✅ Fetched boolean-true columns and dependencies for ${strategy_name}`,
      data: selectedData
    });

  } catch (err) {
    console.error("❌ DB Error:", err.message);
    res.status(500).json({
      error: 'Server Error',
      message: err.message
    });
  }
});



// Start server
app.listen(port, () => {
  console.log(`🚀 Server running on http://localhost:${port}`);
});






// Ensure schema and tables
async function ensureSchemaAndTables() {
  try {
    await pool.query(`CREATE SCHEMA IF NOT EXISTS user_management`);
    await pool.query(`
      CREATE TABLE IF NOT EXISTS user_management.users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        password_hash TEXT NOT NULL,
        api_key TEXT,
        api_secret TEXT
      )
    `);
    await pool.query(`
      CREATE TABLE IF NOT EXISTS user_management.user_strategies (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES user_management.users(id) ON DELETE CASCADE,
        strategy_name VARCHAR(255) NOT NULL
      )
    `);
    console.log("✅ Schema and tables ready");
  } catch (err) {
    console.error("❌ Error creating schema/tables:", err);
    throw err;
  }
}
ensureSchemaAndTables();

// ----------------- STRATEGIES -----------------
app.get("/api/strategies/names", async (req, res) => {
  try {
    const query = `SELECT strategy_name FROM public.strategies_indicators ORDER BY strategy_name`;
    const result = await pool.query(query);
    const strategyNames = result.rows.map(row => row.strategy_name);
    res.json({ data: strategyNames });
  } catch (err) {
    console.error("Error fetching strategies:", err.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ----------------- USERS -----------------

// Register user
app.post("/api/users/register", async (req, res) => {
  const { name, password, apiKey, apiSecret, strategies } = req.body;
  if (!name || !password) return res.status(400).json({ error: "Name and password required" });

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    const insertUserQuery = `
      INSERT INTO user_management.users (name, password_hash, api_key, api_secret)
      VALUES ($1, $2, $3, $4)
      RETURNING id, name, api_key, api_secret
    `;
    const userResult = await pool.query(insertUserQuery, [
      name, hashedPassword, apiKey || null, apiSecret || null
    ]);
    const userId = userResult.rows[0].id;

    // Insert strategies if valid
    if (Array.isArray(strategies) && strategies.length > 0) {
      const insertStrategyQuery = `
        INSERT INTO user_management.user_strategies (user_id, strategy_name)
        VALUES ($1, $2)
      `;
      for (const strat of strategies.filter(Boolean)) {
        await pool.query(insertStrategyQuery, [userId, strat]);
      }
    }

    res.json({ success: true, message: "User and strategies saved successfully", user: userResult.rows[0] });
  } catch (err) {
    console.error("Error registering user:", err.message);
    res.status(500).json({ error: "Internal server error", message: err.message });
  }
});

// Get all users
app.get("/api/users", async (req, res) => {
  try {
    const query = `
      SELECT u.id, u.name, u.api_key, u.api_secret,
             COALESCE(array_agg(us.strategy_name) FILTER (WHERE us.strategy_name IS NOT NULL), '{}') AS strategies
      FROM user_management.users u
      LEFT JOIN user_management.user_strategies us ON u.id = us.user_id
      GROUP BY u.id
      ORDER BY u.name
    `;
    const result = await pool.query(query);
    res.json({ data: result.rows });
  } catch (err) {
    console.error("Error fetching users:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Update user
app.put("/api/users/:id", async (req, res) => {
  const userId = req.params.id;
  const { name, password, apiKey, apiSecret, strategies } = req.body;

  try {
    let fields = [];
    let values = [];
    let idx = 1;

    if (name) { fields.push(`name = $${idx++}`); values.push(name); }
    if (password) { fields.push(`password_hash = $${idx++}`); values.push(await bcrypt.hash(password, 10)); }
    if (apiKey !== undefined) { fields.push(`api_key = $${idx++}`); values.push(apiKey); }
    if (apiSecret !== undefined) { fields.push(`api_secret = $${idx++}`); values.push(apiSecret); }

    if (fields.length > 0) {
      const updateQuery = `UPDATE user_management.users SET ${fields.join(", ")} WHERE id = $${idx}`;
      values.push(userId);
      await pool.query(updateQuery, values);
    }

    // Update strategies
    await pool.query(`DELETE FROM user_management.user_strategies WHERE user_id = $1`, [userId]);
    if (Array.isArray(strategies) && strategies.length > 0) {
      const insertStrategyText = `INSERT INTO user_management.user_strategies (user_id, strategy_name) VALUES ($1, $2)`;
      for (const s of strategies.filter(Boolean)) {
        await pool.query(insertStrategyText, [userId, s]);
      }
    }

    res.json({ success: true, message: "User updated successfully" });
  } catch (err) {
    console.error("Error updating user:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});


















// GET /api/strategies/all_best_tables
app.get('/api/strategies/all_best_tables', async (req, res) => {
  try {
    const exchanges = ['bybit'];
    const symbols = ['BTC', 'ETH', 'SOL'];
    const timeHorizons = ['2h', '6h', '8h'];
    const models = [
      'xg_boost',
      'ridge_regression',
      'random_forest',
      'lstm',
      'linear_regression',
      'knn',
      'gradient_boosting',
      'decision_tree',
      'bayesian_ridge',
      'adaboost'
    ];
    const trialNumbers = Array.from({ length: 50 }, (_, i) => i + 1);

    const allResults = [];

    // Step 1: Generate all table name prefixes
    const tablePrefixes = [];
    exchanges.forEach(exch => {
      symbols.forEach(sym => {
        timeHorizons.forEach(time => {
          models.forEach(model => {
            tablePrefixes.push(`${exch}_${sym}_${time}_${model}_trial`);
          });
        });
      });
    });

    // Step 2: Fetch all table names from the DB once
    const { rows: allTables } = await pool.query(
      `SELECT table_schema, table_name 
       FROM information_schema.tables 
       WHERE table_schema = 'ml_ledger'`
    );

    

    // Step 3: For each prefix, loop only through trial numbers
    for (const prefix of tablePrefixes) {
      let bestTable = null;
      let bestSchema = null;
      let bestPnl = -Infinity;

      for (const trial of trialNumbers) {
        const tableName = `${prefix}${trial}`.toLowerCase();

        // Check if this table exists in our fetched list
        const tableInfo = allTables.find(
          t => t.table_name === tableName
        );
        if (!tableInfo) continue;

        try {
          const schema = tableInfo.table_schema;
          const tbl = tableInfo.table_name;

          // Fetch last pnl_sum
          const lastRow = await pool.query(
            `SELECT pnl_sum FROM "${schema}"."${tbl}" ORDER BY datetime DESC LIMIT 1`
          );

          if (lastRow.rows.length > 0) {
            const pnl = Number(lastRow.rows[0].pnl_sum);
            if (pnl > bestPnl) {
              bestPnl = pnl;
              bestTable = tbl;
              bestSchema = schema;
            }
          }
        } catch (err) {
          console.warn(`⚠️ Could not read table ${tableName}: ${err.message}`);
        }
      }

      // Push the best table data
      if (bestTable && bestSchema) {
        const fullData = await pool.query(
          `SELECT * FROM "${bestSchema}"."${bestTable}" ORDER BY datetime`
        );

        const [exch, sym, time, model] = bestTable.split('_').slice(0, 4);

        allResults.push({
          exchange: exch,
          symbol: sym,
          time_horizon: time,
          model,
          table_name: `${bestSchema}.${bestTable}`,
          max_last_row_pnl_sum: bestPnl,
          data: fullData.rows
        });
      }
    }

    res.json({
      success: true,
      total_found: allResults.length,
      results: allResults
    });

  } catch (error) {
    console.error('❌ Error in /api/strategies/all_best_tables:', error.message);
    res.status(500).json({ error: 'Server error', details: error.message });
  }
});


app.get('/api/strategies/table_data', async (req, res) => {
  try {
    const { table_name } = req.query;
    if (!table_name) return res.status(400).json({ error: "table_name query parameter is required" });

    const tableOnly = table_name.split('.').pop().toLowerCase(); // only table part

    const tableCheck = await pool.query(
      `SELECT table_schema, table_name
       FROM information_schema.tables
       WHERE table_schema = 'ml_ledger'
         AND LOWER(table_name) = $1`,
      [tableOnly]
    );

    if (!tableCheck.rows.length) return res.status(404).json({ error: `Table ${table_name} not found` });

    const { table_schema, table_name: tbl } = tableCheck.rows[0];

    const tableData = await pool.query(
      `SELECT * FROM "${table_schema}"."${tbl}" ORDER BY datetime`
    );

    res.json({ success: true, rows: tableData.rows });
  } catch (err) {
    console.error('❌ Error in /api/strategies/table_data:', err.message);
    res.status(500).json({ error: 'Server error', details: err.message });
  }
});




//last module of live demo trading tables:
// =========================
// Fetch all Bybit Demo records from BTC, ETH, and SOL tables
// =========================
app.get('/api/bybit-demo/all', async (req, res) => {
  console.log("📥 Request for ALL Bybit demo records (BTC, ETH, SOL)");

  try {
    const tables = [
      'public.bybit_demo_btc',
      'public.bybit_demo_eth',
      'public.bybit_demo_sol'
    ];

    let allData = {};

    for (let table of tables) {
      const query = `
        SELECT symbol, position_type, status,
               entry_time, close_time, entry_price, exit_price, quantity,
               tp_price, sl_price, tp_hit, sl_hit, reason,
               balance_before, balance_after, pnl, fee,interval_hours,exchange
        FROM ${table}
        ORDER BY id DESC;
      `;

      const result = await pool.query(query);
      allData[table.split('.').pop()] = result.rows;
    }

    // If all tables are empty
    if (Object.values(allData).every(arr => arr.length === 0)) {
      return res.status(404).json({ error: 'No records found in demo tables.' });
    }

    res.json({
      debug: `✅ Fetched data from ${tables.length} demo tables`,
      data: allData
    });

  } catch (err) {
    console.error("❌ DB Error:", err.message);
    res.status(500).json({
      error: 'Server Error',
      message: err.message
    });
  }
});

