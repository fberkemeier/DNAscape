"""Core symbolic mappings and shared constants for DNAscape."""


class NameMapping:
    def __init__(self):
        self.long_from_short = {
            'ir':'initiation_rate',
            'eff':'efficiency',
            'rt':'replication_timing',
            'iod':'inter_origin_distances',
            'rfd':'fork_directionality',
            'fs':'fork_speed',
            'fss':'fork_speed',
            'nori':'num_oris',
            'ex':'example',
            'rg':'region'
        }
        self.short_from_long = {v:k for k,v in self.long_from_short.items()}
    
    def __getitem__(self, key):
        from ._terminology import canonical_rate_key
        key = str(key)
        key = canonical_rate_key(key)
        if key in self.long_from_short:   # short → long
            return self.long_from_short[key]
        if key in self.short_from_long:   # long → long (canonical)
            return key
        raise KeyError(f"Unknown key: {key}")

mapt = NameMapping()
