#/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pydantic import BaseModel, StrictBool, model_validator 
from pydantic import PositiveInt, PositiveFloat
from pydantic import DirectoryPath, FilePath
from typing import Literal, Self
from pathlib import Path
import subprocess
import yaml

class ExperimentSettings(BaseModel):

    """ Dataclass for experiment settings. """

    # main parameters
    dset: str
    dsrc: Literal["ts","df","gf"]
    arch: Literal["nn","rnn","cnn","res"]
    pret: StrictBool
    pret_mode: StrictBool

    # model parameters
    wdw_len: PositiveInt
    wdw_str: PositiveInt
    str_sts: StrictBool
    enc_feats: PositiveInt
    dec_feats: PositiveInt

    # data parameters
    rho_dfs: PositiveFloat
    num_med: PositiveInt
    nsamp_tra: int
    nsamp_tst: int
    nsamp_pre: int
    val_size: float
    cv_rep: int
    seed: int
   
    # training parameters
    batch_size: PositiveInt
    max_epochs: PositiveInt
    lr: PositiveFloat

    # directories / files
    dir_results: DirectoryPath
    dir_encoders: DirectoryPath
    dir_datasets: DirectoryPath
    dir_training: DirectoryPath
    
    @model_validator(mode="after")
    def valid_arch(self) -> Self:
        valid_combinations = {"ts": ["nn", "rnn", "cnn", "res"], 
            "df": ["cnn", "res"], "gf": ["cnn", "res"]} 
        if self.arch not in valid_combinations[self.dsrc]:
            raise ValueError(f"Architecture {self.arch} not available for '{self.dsrc}'.")
        return self
    
    @model_validator(mode="after")
    def valid_pret_mode(self) -> Self:
        check1 = (self.pret and self.dsrc not in ["df", "gf"])
        check2 = (self.pret_mode and self.dsrc not in ["df", "gf"])
        if check1 or check2:
            raise ValueError("Pretrain only available for dsrc={'df','gf'}.")
        if self.pret and self.pret_mode:
            raise ValueError("'pret_mode' is a pre-step to 'pret'")
        if self.pret and not self.check_encoder():
            raise ValueError("Encoder not found. Please run 'pret_mode' first.")
        if self.pret_mode and self.check_encoder():
            raise ValueError("Encoder for current experiment already exists, program stopped.")
        return self
    
    @model_validator(mode="after")
    def valid_ts(self) -> Self:
        if self.dsrc == "ts" and self.str_sts:
            raise ValueError("If dsrc='ts''str_sts' must be False.")
        return self
    
    def get_experiment_name(self) -> str:
        """ Generates the experiment name. """
        name = f"{self.dset}_{self.dsrc}_{self.arch}_p{self.pret}_pm{self.pret_mode}"
        name += f"wl{self.wdw_len}_ws{self.wdw_str}_ss{int(self.str_sts)}"
        return name

    def get_encoder_name(self) -> str:
        name = f"{self.dset}_{self.dsrc}_{self.arch}"
        name += f"wl{self.wdw_len}_ws{self.wdw_str}_ss{int(self.str_sts)}"
        name += ".pt"
        return name
    
    def get_encoder_path(self) -> Path:
        return (self.dir_encoders / self.get_encoder_name()).absolute()
    
    def check_encoder(self) -> bool:
        """ Checks if there is a pretrained encoder for the experiment. """
        return self.get_encoder_path().exists()
        
class SlurmSettings(BaseModel):

    """ Dataclass for SLURM parameters"""
    
    time: str                   # time string
    account: str                # account string
    partition: str              # partiton string
    cpu: PositiveInt            # number of cpus
    mem: PositiveInt            # number of ram GBs
    gres: str                   # resource string
    constraint: str             # constraing string
    email: str                  # email address
    env_script: FilePath        # path to the enviroment script
    cli_script: FilePath        # path to the CLI script
    modules: list[str]          # list of modules to load
    jobs_dir: DirectoryPath     # directory to store job files
    logs_dir: DirectoryPath     # directory to store output/error files

    @staticmethod
    def load_from_yaml(fname: Path) -> SlurmSettings:
        """Loads settins from YAML file. """
        with open(fname, 'r') as file:
            params = yaml.safe_load(file)
        return SlurmSettings(**params)
    
    def generate_command(self, es: ExperimentSettings) -> str:
        """ Generates the CLI command for the given experiment. """
        params: list[str] = ["dset", "dsrc", "arch", "pret", "pret_mode",
            "wdw_len", "wdw_str", "str_sts", "enc_feats", "dec_feats",
            "rho_dfs", "num_med", "nsamp_tra", "nsamp_tst", "nsamp_pre",
            "val_size", "cv_rep", "seed", "batch_size", "max_epochs", "lr",
            "dir_results", "dir_encoders", "dir_datasets", "dir_training"]
        flags: list[str] = ["pret", "pret_mode", "sts_str"]
        command = f"python {str(self.cli_script)} "
        for var in params:
            if var not in flags:
                command += f"--{var} {str(es.__getattribute__(var))} "
            else:
                command += f"--{var} "
        return command
    
    def launch_experiment(self, es: ExperimentSettings, 
            dry_run: bool = False) -> None:

        name = es.get_experiment_name()
        command = self.generate_command(es)

        # Ensure folders exist
        for folder in [self.jobs_dir, self.logs_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        # Define input / output files
        job_file = self.jobs_dir / (name + ".job")
        out_file = self.logs_dir / (name + ".out")
        err_file = self.logs_dir / (name + ".err")

        # Create SLURM file
        with job_file.open(mode="w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={name}\n")
            if self.account is not None:
                f.write(f"#SBATCH --account={self.account}\n")
            if self.time is not None:
                f.write(f"#SBATCH --time={self.time}\n")
            f.write(f"#SBATCH --partition={self.partition}\n")
            if self.gres is not None:
                f.write(f"#SBATCH --gres={self.gres}\n")
            if self.constraint is not None:
                f.write(f"#SBATCH --constraint={self.constraint}\n")
            f.write(f"#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --ntasks-per-node=1\n")
            f.write(f"#SBATCH --cpus-per-task={str(self.cpu)}\n")
            f.write(f"#SBATCH --mem={str(self.mem)}G\n")
            if self.email is not None:
                f.write(f"#SBATCH --mail-type=END\n")
                f.write(f"#SBATCH --mail-user={self.email}\n")
            f.write(f"#SBATCH -o {out_file}\n")
            f.write(f"#SBATCH -e {err_file}\n")
            for module in self.modules:
                f.write(f"module load {module}\n")
            f.write(f"source {str(self.env_script)}\n")
            f.write(command)
        if not dry_run:
            subprocess.run(["sbatch", str(job_file)])

