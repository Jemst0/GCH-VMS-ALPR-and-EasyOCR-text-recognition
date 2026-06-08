import traceback
import sys
import os


sys.path.insert(0, os.path.dirname(__file__))


def main():
    try:
        try:
            import backend.lpr_infer as li
        except Exception:
            import lpr_infer as li

        args = li.build_args()
        print("pretrained:", args.pretrained)
        print("img_size:", args.img_size)

        model_device = li.load_model(args)
        if model_device:
            model, device = model_device
            print("Loaded model on device:", device)
            try:
                import numpy as np
                import torch

                dummy = np.zeros((args.img_size[1], args.img_size[0], 3), dtype="uint8")
                t = li.numpy2tensor(dummy, args.img_size).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(t)
                print("Forward OK, output shape:", getattr(out, 'shape', None))
            except Exception as ex:
                print("Forward failed:")
                traceback.print_exc()
        else:
            print("load_model returned None")

    except Exception:
        print("ERROR during import/load:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
