"""Fit wavelength-independent Sérsic morphology to candidate cutouts."""

from argparse import ArgumentParser
from pathlib import Path

from astropy.io import fits

from gordian import Gordian
from gordian.config import BandConfig, FitsSource, FitConfig, GordianConfig, IOConfig, ModelConfig


LAPIS_DIR = Path("/Users/Jonah/PhD/Research/lapis")
CUTOUT_BASE = LAPIS_DIR / "data_processed" / "cutouts"
PSF_BASE = LAPIS_DIR / "data_processed" / "filter_psfs"
OUT_BASE = LAPIS_DIR / "outputs" / "gordian_outputs"
FIG_BASE = LAPIS_DIR / "figures" / "gordian_outputs"

def parse_args():
    """Select candidate directories and fitting options."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sightline", type=str, default="lwf")
    parser.add_argument("--objects", nargs="*", default=["ID001", "ID002", "ID003", "ID004"])
    parser.add_argument("--psf-dir", type=Path)
    parser.add_argument("--psf-module", choices=["A", "B"], default="A")
    parser.add_argument("--exclude-bands", nargs="+", default=[])
    parser.add_argument("--size", default=21, type=int)
    parser.add_argument("--fit-type", choices=["independent", "simultaneous"], default="independent")
    parser.add_argument("--method", choices=["mcmc", "laplace", "svi-mvn", "svi-flow"], default="mcmc")
    parser.add_argument("--sky-type", choices=["none", "flat", "tilted-plane"], default="flat")
    return parser.parse_args()


def build_config(object_name, args):
    """Build a GordianConfig for a single candidate cutout directory."""

    cutout_dir = CUTOUT_BASE / args.sightline / object_name
    psf_dir = PSF_BASE / args.sightline
    
    cutout_files = sorted(cutout_dir.glob(f"{object_name}_F*_10arcsec.fits"))
    band_config = {}
    for cutout_file in cutout_files:
        
        filt = cutout_file.stem[len(object_name) + 1:].removesuffix("_10arcsec")
        if filt in args.exclude_bands:
            continue
        if cutout_file.stat().st_size == 0:
            print(f"Skipping {filt}: empty cutout {cutout_file}")
            continue
        psf_file = PSF_BASE / args.sightline / f"mpsf_{filt}.fits"
        if not psf_file.exists():
            if args.psf_module is None:
                raise ValueError(f"No unsuffixed PSF for {filt} in {psf_dir}; select --psf-module A or B.")
            psf_file = psf_dir / f"mpsf_{filt}{args.psf_module}.fits"
        if not psf_file.is_file():
            raise FileNotFoundError(psf_file)
        
        # Check the FITS selections
        with fits.open(cutout_file) as hdul:
            shape = hdul["SCI"].shape
            if hdul["ERR"].shape != shape:
                raise ValueError(f"SCI and ERR shapes differ in {cutout_file}")
            if not 0 < args.size <= min(shape):
                raise ValueError(f"--size must be positive and fit inside {cutout_file}: {shape}")

        band_config[filt] = BandConfig(
            data=FitsSource(cutout_file, hdu="SCI"), 
            err=FitsSource(cutout_file, hdu="ERR"),
            psf=FitsSource(psf_file, hdu=0), 
            sedpy_filter=f"jwst_{filt.lower()}",
            width=args.size, height=args.size,
        )
    if not band_config:
        raise ValueError(f"No selected <name>_F*_10arcsec.fits cutouts")

    sky_suffix = {"none": "_nosky", "flat": "", "tilted-plane": "_tiltedsky"}[args.sky_type]
    run_name = f"sersic{sky_suffix}"

    config = GordianConfig(
        band_config=band_config, cube_config=None,
        model_config=ModelConfig(
            profile_type="sersic", sky_type=args.sky_type,
            band_prior_dict={
                "r_eff": dict(type="uniform", low=0.5, high=20.0),
                "ellip": dict(type="uniform", low=0.0, high=0.9),
                "n": dict(type="uniform", low=0.65, high=8.0),
            },
            const_params=["xc", "yc", "r_eff", "ellip", "theta", "n"], linked_params=[],
        ),
        fit_config=FitConfig(
            fit_type=args.fit_type, loss_func="student_t_loss_free_sys", method=args.method,
            multifitter="poly", multifitter_kwargs={"poly_order": 0},
            mcmc_kwargs={"progress_bar": True, "chain_method": "sequential"},
            num_chains=2, num_warmup=1000, num_samples=1000, seed=67, n_processes=len(band_config), skip_cube_fit=True,
        ),
        io_config=IOConfig(
            out_dir=str(OUT_BASE / args.sightline / object_name / run_name),
            fig_dir=str(FIG_BASE / args.sightline / object_name / run_name), verbose=True, save_observations=True,
        ),
    )
    config.model_config.validate()
    config.fit_config.validate()
    return config


def main():
    """Discover candidates, prepare configurations and fit each object in turn."""
    args = parse_args()

    # Create object-specific directories
    for name in args.objects:
        out_dir = OUT_BASE / args.sightline / name
        fig_dir = FIG_BASE / args.sightline / name
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate every candidate before starting the first fit.
    configs = [build_config(name, args) for name in dict.fromkeys(args.objects)]
    for config in configs:
        print(f"Fitting {config.io_config.out_dir}: {', '.join(config.band_config)}", flush=True)
        Gordian(config).run()


if __name__ == "__main__":
    main()
