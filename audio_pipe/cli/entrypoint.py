import pathlib

import click
import librosa
import numpy as np

from audio_pipe import components, pipeline


@click.group()
@click.option('-p', '--path', type=click.Path(exists=True), default='config.cfg',
    help='Path to configuration file')
@click.pass_context
def cli(ctx, path):
    ctx.ensure_object(dict)
    ctx.obj['path'] = path


@cli.command()
@click.pass_context
def list_components(ctx):
    text = 'Available components'
    click.echo(f'\n{text}\n{"-"*len(text)}')
    for c in components._available:
        click.echo(f'\t- {c}')
    click.echo('\n')


@cli.command()
@click.option('-i', '--inpath', type=click.Path(exists=True),
    help='Path to input directory')
@click.option('-o', '--outpath', type=click.Path(exists=False),
    help='Path to output directory')
@click.pass_context
def run(ctx, inpath, outpath):
    inpath = pathlib.Path(inpath)
    outpath = pathlib.Path(outpath)
    outpath.mkdir(exist_ok=True)

    pipes = pipeline.load(path=ctx.obj['path'])

    with click.progressbar(list(inpath.rglob('*.wav')),
                           label='Preparing features') as files:
        for f in files:
            outdir = outpath.joinpath(f.parent.name)
            outdir.mkdir(exist_ok=True)

            y, sr = librosa.load(f)

            X = y.reshape(1, -1)
            for pipe in pipes:
                res = pipe(X)
                outfile = f'{f.stem}_{pipe.name}.npy'
                np.save(outdir.joinpath(outfile), res, allow_pickle=False)


if __name__ == '__main__':
    cli(obj={})
