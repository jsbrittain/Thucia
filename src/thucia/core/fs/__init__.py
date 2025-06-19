from platformdirs import user_cache_dir

appname = "thucia"
appauthor = "global.Health"
cache_folder = user_cache_dir(appname, appauthor)


def get_cache_folder():
    """Returns the path to the cache folder."""
    return cache_folder
