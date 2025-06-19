from thucia.core import fs
from thucia.flow.wrapper import task


@task
def get_cache_folder(*args, **kwargs):
    return fs.get_cache_folder(*args, **kwargs)
