"""
Script to create betapol.npz containing real-space versions of the
ell-space "T" and "main" beams I took from Yuuki's `fieldlevelbeam` repo

Performs Hankel transforms to convert from ell space to radius space.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0  # Bessel function of the first kind, order 0


def hankel_transform_beam(ell, B_ell, r_arcmin, normalize=True):
    """
    Perform Hankel transform to convert from multipole space to real space.

    For a circularly symmetric beam, the relationship is:
    B(r) = (1/2π) * ∫ B_ell * J_0(ell * r * π/180/60) * ell * d_ell

    where r is in arcminutes and we convert to radians for the Bessel function.

    Parameters:
    -----------
    ell : array
        Multipole values
    B_ell : array
        Beam in multipole space
    r_arcmin : array
        Radial coordinates in arcminutes
    normalize : bool
        Whether to normalize the beam to 1 at r=0

    Returns:
    --------
    array
        Beam profile in real space
    """
    print("Performing Hankel transform...")
    print(f"  ell range: {ell.min()} to {ell.max()}")
    print(f"  r range: {r_arcmin.min():.3f} to {r_arcmin.max():.3f} arcmin")

    # Convert arcminutes to radians for Bessel function
    r_rad = r_arcmin * np.pi / (180 * 60)

    # Initialize output array
    B_r = np.zeros_like(r_arcmin)

    # Perform the integral using trapezoidal rule
    # Skip ell=0 to avoid issues with normalization
    mask = ell > 0
    ell_nonzero = ell[mask]
    B_ell_nonzero = B_ell[mask]

    for i, r in enumerate(r_rad):
        if i % 50 == 0:
            print(f"  Processing r index {i}/{len(r_rad)}")

        # J_0(ell * r) for this radius
        bessel_values = j0(ell_nonzero * r)

        # Integrand: B_ell * J_0(ell * r) * ell
        integrand = B_ell_nonzero * bessel_values * ell_nonzero

        # Integrate using trapezoidal rule
        # Factor of 1/(2π) from the Hankel transform definition
        B_r[i] = np.trapz(integrand, ell_nonzero) / (2 * np.pi)

    if normalize:
        # Normalize to 1 at r=0
        if B_r[0] != 0:
            B_r = B_r / B_r[0]
            print("  Normalized beam peak to 1.0")
        else:
            print("  Warning: Beam is zero at r=0, cannot normalize")

    print(f"  Beam range after transform: {B_r.min():.6f} to {B_r.max():.6f}")
    return B_r


def load_fieldlevel_data():
    """
    Load field-level beam data from the data files.

    Returns:
    --------
    dict
        Dictionary containing ell and beam data for all bands
    """
    # Get the path to the fieldlevelbeam data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "fieldlevelbeam", "data")

    main_beam_file = os.path.join(data_dir, "B_ell_main_beam.npz")
    rc4_beam_file = os.path.join(data_dir, "B_ell_rc4.npz")

    if not os.path.exists(main_beam_file):
        raise FileNotFoundError(f"Main beam file not found: {main_beam_file}")
    if not os.path.exists(rc4_beam_file):
        raise FileNotFoundError(f"RC4 beam file not found: {rc4_beam_file}")

    print("Loading field-level beam data...")
    print(f"  Main beam: {main_beam_file}")
    print(f"  RC4 beam: {rc4_beam_file}")

    # Load main beam data (Bmain)
    main_data = np.load(main_beam_file)
    ell = main_data["ell"]

    # Load RC4 beam data (BT)
    rc4_data = np.load(rc4_beam_file)

    # Verify ell arrays match
    if not np.array_equal(ell, rc4_data["ell"]):
        raise ValueError("ell arrays don't match between main and RC4 beam files")

    data = {"ell": ell, "bands": ["90", "150", "220"], "Bmain": {}, "BT": {}}

    # Extract beam data for each band
    for band in data["bands"]:
        # Main beam uses format B_ell_{band}
        main_key = f"B_ell_{band}"
        if main_key in main_data:
            data["Bmain"][band] = main_data[main_key]
        else:
            raise KeyError(f"Main beam data for {band} GHz not found (key: {main_key})")

        # RC4 beam uses band as key directly
        if band in rc4_data:
            data["BT"][band] = rc4_data[band]
        else:
            raise KeyError(f"RC4 beam data for {band} GHz not found (key: {band})")

    print(f"Loaded data for bands: {data['bands']}")
    print(f"ell range: {ell.min()} to {ell.max()} ({len(ell)} points)")

    return data


def create_betapol_data():
    """
    Create the betapol.npz file containing real-space beam profiles for all bands.
    """
    print("=" * 60)
    print("Creating betapol.npz from field-level beam data")
    print("=" * 60)

    # Load field-level data
    data = load_fieldlevel_data()
    ell = data["ell"]
    bands = data["bands"]

    # Define radial grid for real-space profiles
    # Use a fine grid from 0 to 10 arcmin
    r_arcmin = np.linspace(0, 10, 1000)
    print(f"\nReal-space grid: {len(r_arcmin)} points from 0 to {r_arcmin.max()} arcmin")

    # Initialize output arrays
    output_data = {"r_fine_arcmin": r_arcmin, "bands": bands}

    # Process each band
    for band in bands:
        print("\n" + "=" * 40)
        print(f"Processing {band} GHz")
        print("=" * 40)

        # Get multipole-space data
        Bmain_ell = data["Bmain"][band]
        BT_ell = data["BT"][band]

        print(f"Bmain_ell range: {Bmain_ell.min():.6f} to {Bmain_ell.max():.6f}")
        print(f"BT_ell range: {BT_ell.min():.6f} to {BT_ell.max():.6f}")

        # Perform Hankel transforms
        print("\nTransforming Bmain...")
        Bmain_r = hankel_transform_beam(ell, Bmain_ell, r_arcmin, normalize=True)

        print("\nTransforming BT...")
        BT_r = hankel_transform_beam(ell, BT_ell, r_arcmin, normalize=True)

        # Store results
        output_data[f"Bmain_r_norm_{band}"] = Bmain_r
        output_data[f"BT_r_norm_{band}"] = BT_r

        print(f"\nCompleted {band} GHz:")
        print(f"  Bmain(r): min={Bmain_r.min():.6f}, max={Bmain_r.max():.6f}")
        print(f"  BT(r): min={BT_r.min():.6f}, max={BT_r.max():.6f}")

    # Save output file
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "betapol.npz")

    print("\n" + "=" * 60)
    print(f"Saving betapol data to: {output_file}")
    np.savez(output_file, **output_data)

    print(f"File saved with keys: {list(output_data.keys())}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    return output_file


def plot_beam_profiles(betapol_file):
    """
    Create diagnostic plots of the beam profiles.
    """
    print("\n" + "=" * 60)
    print("Creating diagnostic plots")
    print("=" * 60)

    data = np.load(betapol_file)
    r_arcmin = data["r_fine_arcmin"]
    bands = ["90", "150", "220"]

    # Create subplots for each band
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Betapol Beam Profiles (Real Space)", fontsize=16)

    for i, band in enumerate(bands):
        # Top row: Individual profiles
        ax_profiles = axes[0, i]
        Bmain_r = data[f"Bmain_r_norm_{band}"]
        BT_r = data[f"BT_r_norm_{band}"]

        ax_profiles.plot(r_arcmin, Bmain_r, "b-", label="Bmain", linewidth=2)
        ax_profiles.plot(r_arcmin, BT_r, "r-", label="BT", linewidth=2)
        ax_profiles.set_title(f"{band} GHz")
        ax_profiles.set_xlabel("Radius [arcmin]")
        ax_profiles.set_ylabel("Normalized Amplitude")
        ax_profiles.grid(True, alpha=0.3)
        ax_profiles.legend()
        ax_profiles.set_xlim(0, 5)  # Focus on central region

        # Bottom row: Difference profiles for different beta_pol values
        ax_diff = axes[1, i]

        # Show P beam for different beta_pol values
        beta_vals = [0.0, 0.5, 1.0]
        colors = ["g", "orange", "purple"]

        for beta_pol, color in zip(beta_vals, colors):
            P_beam = Bmain_r + beta_pol * (BT_r - Bmain_r)
            ax_diff.plot(
                r_arcmin,
                P_beam,
                color=color,
                linewidth=2,
                label=f"P beam (β={beta_pol})",
            )

        ax_diff.plot(r_arcmin, BT_r, "r--", alpha=0.7, label="T beam (BT)")
        ax_diff.set_title(f"{band} GHz - P Beam Interpolation")
        ax_diff.set_xlabel("Radius [arcmin]")
        ax_diff.set_ylabel("Normalized Amplitude")
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend()
        ax_diff.set_xlim(0, 5)

    plt.tight_layout()

    # Save plot
    plot_file = betapol_file.replace(".npz", "_profiles.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved diagnostic plot to: {plot_file}")

    # Print summary statistics
    print("\nSummary statistics:")
    for band in bands:
        Bmain_r = data[f"Bmain_r_norm_{band}"]
        BT_r = data[f"BT_r_norm_{band}"]

        print(f"\n{band} GHz:")
        print(f"  Bmain: peak={Bmain_r.max():.6f}, FWHM≈{estimate_fwhm(r_arcmin, Bmain_r):.2f} arcmin")
        print(f"  BT:    peak={BT_r.max():.6f}, FWHM≈{estimate_fwhm(r_arcmin, BT_r):.2f} arcmin")

        # Maximum difference for beta_pol=1
        max_diff = np.max(np.abs(BT_r - Bmain_r))
        print(f"  Max |BT - Bmain|: {max_diff:.6f}")


def estimate_fwhm(r, profile):
    """
    Estimate FWHM by finding where profile drops to half maximum.
    """
    half_max = profile.max() / 2
    idx = np.where(profile <= half_max)[0]
    if len(idx) > 0:
        return r[idx[0]] * 2  # Approximate FWHM
    else:
        return np.nan


if __name__ == "__main__":
    try:
        # Create the betapol data file
        output_file = create_betapol_data()

        # Create diagnostic plots
        plot_beam_profiles(output_file)

        print("\n" + "=" * 60)
        print("SUCCESS: betapol.npz created successfully!")
        print(f"Location: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
