<h1>📈 Monte Carlo Simulation Web App</h1>

<p>This web application provides a frontend interface to perform Monte Carlo simulations on yield curve data sourced from the U.S. Treasury. It utilizes <strong>Gradio</strong> for the user interface, <strong>Pandas</strong> and <strong>NumPy</strong> for data processing, <strong>Plotly</strong> for interactive plotting, and optionally <strong>PyOpenCL</strong> for GPU acceleration of simulations.</p>

<h2>🚀 Features</h2>
<ul>
  <li>Fetch and process the latest Daily Treasury Yield Curve Rates.</li>
  <li>Run Monte Carlo simulations on 10-Year yield curve data.</li>
  <li>GPU acceleration with OpenCL (optional).</li>
  <li>Interactive visualization of simulation results using Plotly.</li>
</ul>

<h2>🔧 Installation</h2>
<p>Clone the repository and install all dependencies using:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<p><em>Note:</em> <code>pyopencl</code> is optional and used for GPU acceleration. If your system supports OpenCL, it will be utilized automatically.</p>

<h2>📄 File Overview</h2>
<ul>
  <li><strong>app.py</strong> — Main application script with Gradio interface and simulation logic.</li>
  <li><strong>requirements.txt</strong> — Python dependencies list.</li>
</ul>

<h2>⚙️ Usage</h2>
<pre><code>python app.py</code></pre>
<p>Then, follow the Gradio URL provided in the terminal to launch the web app in your browser.</p>

<h2>📡 Data Source</h2>
<p>Data is fetched from the U.S. Treasury:</p>
<blockquote><a href="https://home.treasury.gov/resource-center/data-chart-center/interest-rates">https://home.treasury.gov/resource-center/data-chart-center/interest-rates</a></blockquote>

<h2>📜 License</h2>
<p>This project is open for educational and research purposes. Refer to the code comments and data source license for more details.</p>