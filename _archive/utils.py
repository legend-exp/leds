from __future__ import annotations

import pathlib

from legendmeta import LegendMetadata
from legendmeta.catalog import Props


def sorter(path, timestamp, key="String", datatype="cal", spms=False):
    prod_config = str(pathlib.Path(path) / "config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]

    cfg_file = prod_config["paths"]["metadata"]
    configs = LegendMetadata(path=cfg_file)
    chmap = configs.channelmap(timestamp).map("daq.rawid")

    software_config_path = prod_config["paths"]["config"]
    software_config_db = LegendMetadata(path=software_config_path)
    software_config = software_config_db.on(timestamp, system=datatype).analysis

    out_dict = {}
    # SiPMs sorting
    if spms:
        chmap = chmap.map("system", unique=False)["spms"]
        if key == "Barrel":
            mapping = chmap.map("daq.rawid", unique=False)
            for pos in ["top", "bottom"]:
                for barrel in ["IB", "OB"]:
                    out_dict[f"{barrel}-{pos}"] = [
                        k
                        for k, entry in sorted(mapping.items())
                        if barrel in entry["location"]["fiber"]
                        and pos in entry["location"]["position"]
                    ]
        return out_dict, chmap

    out_key = "{key}:{k:02}"
    primary_key = "location.string"
    secondary_key = "location.position"
    mapping = chmap.map(primary_key, unique=False)
    for k, entry in sorted(mapping.items()):
        out_dict[out_key.format(key=key, k=k)] = [
            entry.map(secondary_key)[pos].daq.rawid
            for pos in sorted(entry.map(secondary_key))
            if entry.map(secondary_key)[pos].system == "geds"
        ]

    out_dict = {
        entry: out_dict[entry] for entry in list(out_dict) if len(out_dict[entry]) > 0
    }
    return out_dict, software_config, chmap
