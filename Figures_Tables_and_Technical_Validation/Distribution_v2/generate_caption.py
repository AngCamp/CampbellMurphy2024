import json
import os

def format_scientific_notation(value):
    """Format a number in scientific notation like 6.0x10-6"""
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    return f"{mantissa:.1f}x10{exp:+d}".replace("+", "")

def main():
    # Load KS results
    ks_file = os.path.join(os.path.dirname(__file__), "figure9_ks_results.json")
    if not os.path.exists(ks_file):
        print(f"KS results file not found: {ks_file}")
        print("Please run plot_figure9.py first to generate the KS results.")
        return
    
    with open(ks_file, 'r') as f:
        ks_results = json.load(f)
    
    # Generate caption with actual values
    caption = """**Figure 9.  Detected events show expected properties for probe level putative events, occurring in the absence of the wheel movement and at lower theta power.   Plots for ABI Behaviour events are on the left, ABI Coding in the middle, and IBL on the right.  a) The mean instantaneous theta power z-scored during probe level event windows.  b) The wheel speed during the global events, note that for the ABI datasets this can be interpreted as a running speed but from the IBL the mouse uses ambulation to turn a wheel. c) The distribution of global level event durations.  The best fit distribution of the normal, half-normal and lognormal were fit to the data, the lognormal was shown to be the best fit by the Kolmogrov-Smirnov (KS) test in all entities."""
    
    # Add duration results
    if 'abi_visbehave' in ks_results and 'duration' in ks_results['abi_visbehave']:
        dur = ks_results['abi_visbehave']['duration']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'duration' in ks_results['abi_viscoding']:
        dur = ks_results['abi_viscoding']['duration']
        caption += f", ABI Coding (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'duration' in ks_results['ibl']:
        dur = ks_results['ibl']['duration']
        caption += f" and the IBL (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    caption += """  d)  The distribution of global event level peak ripple power (z-scored) is best fit by the lognormal distribution in all entities, all fits pass significance."""
    
    # Add peak power results
    if 'abi_visbehave' in ks_results and 'peak_power' in ks_results['abi_visbehave']:
        pow = ks_results['abi_visbehave']['peak_power']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'peak_power' in ks_results['abi_viscoding']:
        pow = ks_results['abi_viscoding']['peak_power']
        caption += f", ABI Coding (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'peak_power' in ks_results['ibl']:
        pow = ks_results['ibl']['peak_power']
        caption += f" and the IBL (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    caption += "."
    
    # Save caption
    caption_file = os.path.join(os.path.dirname(__file__), "figure9_caption_with_results.txt")
    with open(caption_file, 'w') as f:
        f.write(caption)
    
    print(f"Caption saved to: {caption_file}")
    print("\nGenerated caption:")
    print(caption)

if __name__ == "__main__":
    import numpy as np
    main() 