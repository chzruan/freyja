# to run the full pipeline (run the rule 'all'):
# snakemake -s Snakefile_GLAMx100.py -n # dry-run
# snakemake -s Snakefile_GLAMx100.py --cores 1
# snakemake -s Snakefile_GLAMx100.py --cores 1 -f 

# to run a rule:
# snakemake -s Snakefile_GLAMx100.py --cores 1 $RULENAME
# snakemake -s Snakefile_GLAMx100.py --cores 1 -f $RULENAME
# snakemake -s Snakefile_GLAMx100.py "data/theory_xiSgcs_GR_LOWZ_z0.25.hdf5" --dag | dot -Tpdf > pipeline_xiS_gg.pdf

CONFIG_PATH = "config_GLAMx100.yaml"
configfile: CONFIG_PATH
gravity     = config["basic"]["gravity"]
dataflag    = config["basic"]["dataflag"]
redshift    = config["basic"]["redshift"]


rule all:
    input:
        xiS_plot = f"plots/theory_xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.pdf",
        xiR_plot = f"plots/xiRgcs_{gravity}_{dataflag}_z{redshift:.2f}.pdf",
        velmom_plot = f"plots/velmom_{gravity}_{dataflag}_z{redshift:.2f}.pdf"



rule clean_data:
    shell:
        "rm -rf data/*{dataflag}*"



# rule populate_HOD:
#     input:
#         f"/cosma8/data/dp203/dc-ruan1/DESI_MGx100/data/{gravity}/halo-catalogues.redshift{redshift:.2f}.hdf5"
#     output: 
#         f"data/HOD_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     shell:
#         "python3  populate.py   \
#         --halo_path     {input} \
#         --gal_path      {output}\
#         --config_path   {CONFIG_PATH}"


# rule measure_xiRgcs:
#     input:
#         f"data/HOD_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     output:
#         f"data/xiRgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     shell:
#         "python3  measure_xi-R-gcs.py \
#         --gal_path      {input} \
#         --output_path   {output} \
#         --config_path   {CONFIG_PATH}"


rule plot_xiRgcs:
    input:
        f"data/xiRgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
    output:
        f"plots/xiRgcs_{gravity}_{dataflag}_z{redshift:.2f}.pdf"
    script:
        "plot_xi-R-gcs.py"


# rule measure_xiSgcs:
#     input:
#         f"data/HOD_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     output:
#         f"data/xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     shell:
#         "python3  measure_xi-S-gcs.py \
#         --gal_path      {input} \
#         --output_path   {output} \
#         --config_path   {CONFIG_PATH}"



# rule measure_velmomgcs:
#     input:
#         f"data/HOD_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     output:
#         f"data/velmom_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
#     shell:
#         "julia  measure_velmom-gcs.jl  {input}  {output} {CONFIG_PATH}"


rule plot_velmomgcs:
    input:
        f"data/velmom_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
    output:
        f"plots/velmom_{gravity}_{dataflag}_z{redshift:.2f}.pdf"
    script:
        "plot_velmom-gcs.py"


rule theory_xiSgcs:
    input:
        xiR = f"data/xiRgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5",
        vm = f"data/velmom_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
    output:
        f"data/theory_xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
    shell:
        "python3  theory_xiS-gcs.py \
        --xiR_path      {input.xiR} \
        --velmom_path   {input.vm}  \
        --output_path   {output}    \
        --config_path   {CONFIG_PATH}"


rule plot_halostr:
    input:
        data   = f"data/xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5",
        theory = f"data/theory_xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.hdf5"
    output:
        pdf = f"plots/theory_xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.pdf",
        png = f"plots/theory_xiSgcs_{gravity}_{dataflag}_z{redshift:.2f}.png"
    script:
        "plot_halostr.py"


