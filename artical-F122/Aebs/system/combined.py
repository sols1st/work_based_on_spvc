"""Evaluate the default VISION-SLS SPVC system; retain explicit cGAN replay."""
import argparse
import runpy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frontend', choices=['vision-sls','cgan'], default='vision-sls')
    entry, remaining = parser.parse_known_args()
    if entry.frontend == 'cgan':
        if remaining:
            parser.error('legacy cGAN replay does not accept extra arguments')
        runpy.run_module('Aebs.system.combined_legacy', run_name='__main__')
    else:
        from Aebs.vision_sls.evaluate import main as evaluate
        evaluate(remaining)


if __name__ == '__main__':
    main()
