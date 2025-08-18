import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cli", "api"],
        default="cli",
        help="Mode to run the application",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="Input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Input prompt",
    )
    args = parser.parse_args()

    if args.mode == "api":
        import uvicorn
        from server import app

        uvicorn.run(app, host="0.0.0.0", port=8186)
    else:
        if (not args.image) or (not args.prompt):
            print("image and prompt are both required")

        from vl.detection import QwenVLDetection

        det = QwenVLDetection()
        det.detect(image=args.image, target=args.prompt)
