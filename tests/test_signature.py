import pytest

import torch
import pygsig.signature as sig
import signatory


def test_import_signatory():
    assert signatory is not None

if __name__ == "__main__":
    test_import_signatory()
    print("Signatory imported successfully")

def signature_linear_layer():
    layer = sig.SignatureLinear(in_channels=2,out_channels=3,depth=3,bias=False)

    with torch.no_grad():
        W = torch.tensor([[3, 2],[2, 1],[3, 0]]).float()
        layer.linear[0].weight.data = W
        layer.linear[1].weight.data = W
        layer.linear[2].weight.data = W

        input = torch.tensor([ 1,  0, 1, -1, 1,  0,  0, 1,  0, -1,  0, -1,  0,  0]).float()


        input_1 = signatory.extract_signature_term(input,channels=2,depth=1)
        input_2 = signatory.extract_signature_term(input,channels=2,depth=2).reshape(2,2)
        input_3 = signatory.extract_signature_term(input,channels=2,depth=3).reshape(2,2,2)

        output = layer(input)

    output_1 = torch.matmul(W,input_1)
    output_2 = torch.matmul(torch.matmul(W,input_2),W.T)
    output_3 = torch.einsum('def,ad,be,cf->abc',input_3,W,W,W)

    assert torch.all(output == torch.concat([output_1,output_2.reshape(9),output_3.reshape(27)]))
