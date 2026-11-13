import argparse


def main():
    p = argparse.ArgumentParser(description='Soccer ball tracking mini-project')
    sub = p.add_subparsers(dest='command', required=True)
    from . import download
    download.parser(sub)
    # Modules register small, independent commands.
    for name in ['data', 'test_clip', 'annotate', 'pipeline', 'nn', 'evaluate']:
        module = __import__('balltrack.' + name, fromlist=['parser'])
        module.parser(sub)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
