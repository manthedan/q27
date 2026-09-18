#!/usr/bin/env python3
"""Model-free source/rotation/codec contracts; run directly or with python -O."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np

import repack_bonsai2 as b2
from repack import QWEN35_REQUIRED_METADATA, repack_t2, to_f32
from prism_gguf import register_type


class Field:
    def __init__(self,value): self.value=value
    def contents(self): return self.value


def fixture():
    meta = {**QWEN35_REQUIRED_METADATA, "general.architecture":"qwen35",
            "general.name":"Hf", "qwen35.block_count":64,
            "prism.hadamard.version":1, "prism.hadamard.block_size":1024,
            "prism.hadamard.transform":"normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis":"input-last-dimension", "prism.hadamard.sign_mode":"explicit",
            "prism.hadamard.sign_widths":[5120,6144,17408],
            "prism.hadamard.sign_values":[1,-1]*14336,
            "prism.hadamard.inverse_weight_names":["token_embd.weight"],
            "prism.hadamard.gdn_v_grouped":True, "tokenizer.chat_template":"fixture"}
    meta["prism.hadamard.weight_names"] = [n for n,(dt,_) in b2.SPECS.items() if dt=="PQ2_0" and n!="token_embd.weight"]
    tensors=[SimpleNamespace(name=n,tensor_type=SimpleNamespace(name=dt),shape=list(reversed(shape)))
             for n,(dt,shape) in b2.SPECS.items()]
    return SimpleNamespace(fields={k:Field(v) for k,v in meta.items()},tensors=tensors)


class Contracts(unittest.TestCase):
    def test_manifest(self):
        reader=fixture()
        result=b2.source_metadata(reader)
        self.assertEqual(result["quant_policy"],b2.POLICY)
        self.assertEqual(result["general.name"],"Hf")
        self.assertEqual(result["q27.model_profile"],b2.PROFILE)
        self.assertEqual(len(result["prism.hadamard.weight_names"]),401)
        for mutate in (lambda r:r.tensors.pop(),lambda r:r.tensors.append(r.tensors[0]),
                       lambda r:setattr(r.tensors[0],"shape",[128,128]),
                       lambda r:setattr(r.tensors[0],"tensor_type",SimpleNamespace(name="PTQ1_0"))):
            bad=copy.deepcopy(reader);mutate(bad)
            with self.assertRaises(ValueError):b2.source_metadata(bad)

    def test_metadata_fail_closed(self):
        reader=fixture()
        for key in b2.ROTATION_FIELDS:
            bad=copy.deepcopy(reader);del bad.fields[key]
            with self.assertRaises(ValueError,msg=key):b2.source_metadata(bad)
        mutations={"version":2,"block_size":1024.0,"transform":"other",
                   "sign_mode":"identity","gdn_v_grouped":1,"inverse_weight_names":[],
                   "sign_widths":[5120,5120,17408],"weight_names":["output.weight"],
                   "sign_values":[True]*28672,"unknown":0}
        for key,value in mutations.items():
            bad=copy.deepcopy(reader);bad.fields[b2.PREFIX+key]=Field(value)
            with self.assertRaises(ValueError,msg=key):b2.source_metadata(bad)
        bad=copy.deepcopy(reader);bad.fields[b2.PREFIX+"sign_values"].value[2]=0
        with self.assertRaises(ValueError):b2.source_metadata(bad)
        bad=copy.deepcopy(reader);bad.fields[b2.PREFIX+"weight_names"].value.append("output.weight")
        with self.assertRaises(ValueError):b2.source_metadata(bad)

    def test_pq2_bytes(self):
        rng=np.random.default_rng(123)
        codes=rng.integers(0,3,(3,256),dtype=np.uint8)
        scales=np.array([0.125,0.25,0.5,1,2,4],dtype="<f2")
        packed=(codes[:,0::4] | codes[:,1::4]<<2 | codes[:,2::4]<<4 | codes[:,3::4]<<6).reshape(6,32)
        blocks=np.concatenate([scales.view(np.uint8).reshape(6,2),packed],axis=1)
        t=SimpleNamespace(name="fixture",shape=[256,3],data=blocks,tensor_type=SimpleNamespace(name="PQ2_0"))
        data,out_scales,zero=repack_t2(t)
        self.assertEqual(data,packed.tobytes());self.assertEqual(out_scales,scales.tobytes())
        self.assertEqual(zero,float(np.mean(codes==1)))
        t.data=blocks.copy();t.data[0,2]|=3
        with self.assertRaises(ValueError):repack_t2(t)
        t.shape=[64,6]
        with self.assertRaises(ValueError):repack_t2(t)

    def test_bf16_exact(self):
        bits=np.array([0x0001,0x8001,0x3f81,0xbf81,0x0000,0x8000],dtype=np.uint16)
        t=SimpleNamespace(name="gate",shape=[3,2],data=bits,tensor_type=SimpleNamespace(name="BF16"))
        widened=to_f32(t)
        np.testing.assert_array_equal(widened.view(np.uint32).reshape(-1),bits.astype(np.uint32)<<16)

    def test_type_collision(self):
        with self.assertRaises(ValueError):register_type("Q2_0",42,64,18)
        with self.assertRaises(ValueError):register_type("OTHER",142,128,34)
        self.assertEqual(register_type("PQ2_0",142,128,34).value,142)


if __name__=="__main__":unittest.main()
