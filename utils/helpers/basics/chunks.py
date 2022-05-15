def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    From https://stackoverflow.com/a/312464/8162243.

    Args:
        lst (list): Input list.
        n (int): Chunk Size.

    Yields:
        list: The next chunk of sliced up list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
