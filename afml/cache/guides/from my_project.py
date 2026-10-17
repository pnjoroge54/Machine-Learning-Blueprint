from my_project.cache import robust_cacheable


@robust_cacheable
def my_function(df):
    # Expensive computation
    return result
