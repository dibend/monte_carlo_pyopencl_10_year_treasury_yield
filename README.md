<img src="https://github.com/dibend/monte_carlo_pyopencl_10_year_treasury_yield/blob/main/10y-treasury-yield.png?raw=true" width="1000" height="700">
<h1>📊 Monte Carlo Yield Curve Simulator</h1>

<p>This application performs interactive Monte Carlo simulations on U.S. Treasury 10-Year yield curve data. It uses <strong>Gradio</strong> for the user interface and <strong>Plotly</strong> for visualization. Optionally, it can leverage <strong>PyOpenCL</strong> for GPU acceleration of simulations.</p>

<h2>🌐 Data Source</h2>
<p>This app fetches daily Treasury par yield curve rates from:</p>
<pre><code>https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2024/all?type=daily_treasury_yield_curve&amp;field_tdr_date_value_month=202406&amp;page&amp;_format=csv</code></pre>
<p>It focuses on the <code>10 Yr</code> maturity series.</p>

<h2>🛠 Features</h2>
<ul>
  <li>Dynamic plotting of simulated interest rate paths.</li>
  <li>Interactive configuration: number of simulations and forecast horizon.</li>
  <li>Automatic device detection via OpenCL (fallback to NumPy if GPU unavailable).</li>
  <li>Memory safety checks to prevent excessive allocations on GPU.</li>
</ul>

<h2>📦 Installation</h2>
<p>Install the dependencies using:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<p><strong>Note:</strong> <code>pyopencl</code> is optional. If not installed, the app defaults to CPU (NumPy).</p>

<h2>▶️ Usage</h2>
<pre><code>python app.py</code></pre>
<p>The app will start a Gradio web interface in your browser.</p>

<h2>🧪 Requirements File</h2>
<p>The <code>requirements.txt</code> should include:</p>
<pre><code>gradio
pandas
numpy
plotly
pyopencl  # Optional, only needed for GPU acceleration
</code></pre>

<h2>📈 Output</h2>
<ul>
  <li>Up to 500,000 simulation paths</li>
  <li>Project up to 3 years (1095 days)</li>
  <li>Includes confidence intervals and mean projections</li>
</ul>

<h2>🔒 License</h2>
<p>This project is intended for research and educational use. Please respect the U.S. Treasury data terms of use.</p>
