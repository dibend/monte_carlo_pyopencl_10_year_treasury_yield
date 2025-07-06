import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# Try to import pyopencl, but don't fail if it's not installed.
# We will handle the case where it's unavailable.
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    print("Warning: PyOpenCL is not installed. GPU acceleration will be disabled.")


# --- Configuration ---
# URL for the Daily Treasury Par Yield Curve Rates from the US Treasury website.
DATA_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2024/all?type=daily_treasury_yield_curve&field_tdr_date_value_month=202406&page&_format=csv"
SERIES_TO_SIMULATE = '10 Yr'

# --- OpenCL Kernel Definition ---
# This is the C code that will be executed on the GPU/CPU device.
# It calculates one full simulation path per work-item.
OPENCL_KERNEL = """
__kernel void run_simulation(
    const int num_days,
    const double drift,
    const double volatility,
    const double start_price,
    __global const double* random_shocks,
    __global double* simulations)
{
    int i = get_global_id(0); // Get the unique ID for this simulation path

    // Each work-item handles one simulation path.
    // The random_shocks are laid out as [day0_sim0, day0_sim1, ..., day1_sim0, day1_sim1, ...]
    int num_simulations = get_global_size(0);
    
    // Set the starting price for this path
    simulations[i] = start_price;

    double current_price = start_price;

    for (int t = 0; t < num_days; ++t) {
        // Calculate the index for the random shock for the current day and simulation path
        int shock_index = t * num_simulations + i;
        double price_movement = exp(drift - 0.5 * volatility * volatility + volatility * random_shocks[shock_index]);
        current_price *= price_movement;
        
        // Ensure rate doesn't go below zero
        if (current_price < 0.0) {
            current_price = 0.0;
        }

        // Store the result for this day and path
        // The output simulations array is laid out similarly to the input shocks
        simulations[(t + 1) * num_simulations + i] = current_price;
    }
}
"""

# --- Core Functions ---

def fetch_data_from_csv(url):
    """Fetches historical treasury data from a given CSV URL."""
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        if SERIES_TO_SIMULATE not in df.columns:
            raise gr.Error(f"Series '{SERIES_TO_SIMULATE}' not in CSV. Available: {', '.join(df.columns)}")

        data = df[SERIES_TO_SIMULATE].ffill().dropna()
        return data
    except Exception as e:
        raise gr.Error(f"Failed to process CSV. Error: {e}")

def get_opencl_devices():
    """Enumerates available OpenCL devices."""
    devices = ["CPU (Numpy)"]
    if not PYOPENCL_AVAILABLE:
        return devices
    try:
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                devices.append(f"{platform.name} :: {device.name}")
    except Exception as e:
        print(f"Warning: Could not enumerate OpenCL devices: {e}")
    return devices

def get_cl_device_from_string(device_string):
    """Gets the PyOpenCL device object from its string name."""
    if not PYOPENCL_AVAILABLE or device_string == "CPU (Numpy)":
        return None
    try:
        platform_name_str, device_name_str = [s.strip() for s in device_string.split('::', 1)]
        for p in cl.get_platforms():
            if platform_name_str in p.name:
                for d in p.get_devices():
                    if device_name_str in d.name:
                        return d
    except Exception as e:
        print(f"Could not find specified OpenCL device: {device_string}. Error: {e}")
    return None

def run_monte_carlo_simulation_numpy(historical_data, num_simulations, num_days):
    """Original Numpy-based Monte Carlo simulation for CPU."""
    log_returns = np.log(1 + historical_data.pct_change()).dropna()
    drift = log_returns.mean()
    volatility = log_returns.std()
    start_price = historical_data.iloc[-1]
    
    simulations = np.zeros((num_days + 1, num_simulations))
    simulations[0] = start_price

    for i in range(num_simulations):
        random_shocks = np.random.normal(loc=0, scale=1, size=num_days)
        daily_movement = np.exp(drift - 0.5 * (volatility**2) + volatility * random_shocks)
        simulations[1:, i] = start_price * daily_movement.cumprod()
        simulations[simulations[:, i] < 0, i] = 0
    return simulations

def run_monte_carlo_simulation_opencl(historical_data, num_simulations, num_days, device):
    """OpenCL-accelerated Monte Carlo simulation."""
    # --- PRE-COMPUTATION MEMORY CHECK ---
    # Check if the requested buffer size will exceed device limits before trying to allocate.
    max_alloc_size = device.max_mem_alloc_size
    # We need two main buffers: one for shocks and one for the results.
    # float64 is 8 bytes.
    required_mem = (num_days * num_simulations * 8) + ((num_days + 1) * num_simulations * 8)

    if required_mem > max_alloc_size:
        raise gr.Error(
            f"Memory Error: Requested simulation size ({required_mem / 1024**2:.2f} MB) "
            f"exceeds the device's maximum allocation size ({max_alloc_size / 1024**2:.2f} MB). "
            "Please reduce the number of simulations/days or use the CPU (Numpy) option."
        )
        
    log_returns = np.log(1 + historical_data.pct_change()).dropna()
    drift = log_returns.mean()
    volatility = log_returns.std()
    start_price = historical_data.iloc[-1]

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    random_shocks = np.random.normal(size=num_days * num_simulations).astype(np.float64)
    simulations_host = np.zeros(shape=((num_days + 1) * num_simulations), dtype=np.float64)
    
    mf = cl.mem_flags
    random_shocks_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=random_shocks)
    simulations_g = cl.Buffer(ctx, mf.WRITE_ONLY, simulations_host.nbytes)
    
    prg = cl.Program(ctx, OPENCL_KERNEL).build()
    prg.run_simulation(
        queue, (num_simulations,), None, np.int32(num_days), np.float64(drift),
        np.float64(volatility), np.float64(start_price), random_shocks_g, simulations_g
    )
    
    cl.enqueue_copy(queue, simulations_host, simulations_g).wait()
    return simulations_host.reshape(num_days + 1, num_simulations)

def plot_simulation(historical_data, simulations, num_days):
    """Plots the historical data and simulation results using Plotly."""
    fig = go.Figure()
    
    last_historical_date = historical_data.index[-1]
    forecast_dates = pd.date_range(start=last_historical_date, periods=num_days + 1).tolist()

    num_to_plot = min(simulations.shape[1], 100)
    for i in range(num_to_plot):
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=simulations[:, i], mode='lines',
            line=dict(width=0.5, color='rgba(0, 0, 255, 0.2)'),
            showlegend=False, hoverinfo='none'
        ))

    p5 = np.percentile(simulations, 5, axis=1)
    p95 = np.percentile(simulations, 95, axis=1)
    mean = np.mean(simulations, axis=1)
    
    fig.add_trace(go.Scatter(x=forecast_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=p5, mode='lines', line=dict(width=0),
        fillcolor='rgba(255, 165, 0, 0.3)', fill='tonexty',
        name='90% Confidence Interval', hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=historical_data.index, y=historical_data, mode='lines', 
        name=f'Historical {SERIES_TO_SIMULATE} Yield', line=dict(color='black', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mean, mode='lines', name='Mean Simulation Path',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f"Monte Carlo Simulation of {SERIES_TO_SIMULATE} Treasury Yield ({simulations.shape[1]} Paths)",
        xaxis_title="Date", yaxis_title="Interest Rate (%)",
        legend_title="Legend", template="plotly_white"
    )
    
    return fig

# --- Gradio Interface Function ---
def generate_simulation_plot(device_string, num_simulations, num_days):
    """Main function called by the Gradio interface."""
    num_simulations = int(num_simulations)
    num_days = int(num_days)

    historical_data = fetch_data_from_csv(DATA_URL)
    device = get_cl_device_from_string(device_string)

    if device:
        print(f"Running simulation on OpenCL device: {device_string}")
        simulations = run_monte_carlo_simulation_opencl(historical_data, num_simulations, num_days, device)
    else:
        print("Running simulation on CPU with Numpy.")
        simulations = run_monte_carlo_simulation_numpy(historical_data, num_simulations, num_days)
    
    fig = plot_simulation(historical_data, simulations, num_days)
    return fig

# --- Build and Launch Gradio App ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Monte Carlo Simulation with OpenCL Acceleration and Plotly")
    gr.Markdown(
        f"This tool runs a Monte Carlo simulation on the **{SERIES_TO_SIMULATE} Treasury Yield** using data from a `home.treasury.gov` CSV link. "
        "You can select your GPU or CPU via OpenCL for accelerated computation. The plot is interactive."
    )

    with gr.Row():
        with gr.Column(scale=1):
            device_selector = gr.Dropdown(choices=get_opencl_devices(), value=get_opencl_devices()[0], label="Select Computation Device")
            # UPDATED: Lowered the max to a more reasonable value to avoid instant memory errors.
            num_simulations_slider = gr.Slider(minimum=250, maximum=500000, value=5000, step=250, label="Number of Simulations")
            num_days_slider = gr.Slider(minimum=30, maximum=1095, value=365, step=5, label="Number of Days to Project")
            run_button = gr.Button("Run Simulation", variant="primary")

        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Simulation Results")

    run_button.click(
        fn=generate_simulation_plot,
        inputs=[device_selector, num_simulations_slider, num_days_slider],
        outputs=plot_output
    )

if __name__ == "__main__":
    if not PYOPENCL_AVAILABLE:
        print("\nNOTE: PyOpenCL is not installed. The device selector will only show the 'CPU (Numpy)' option.")
        print("To enable GPU acceleration, please install it using: pip install pyopencl\n")
    # ADDED: share=True and debug=True as requested by user's traceback.
    demo.launch(share=True, debug=True)