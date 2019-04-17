import io

def visualize_html(tokens, probs, color=(0, 255, 0, .5)):
    """len(tokens) == len(probs)"""
    red, green, blue, alpha = color
    strio = io.StringIO()
    for tok, prob in zip(tokens, probs):
        r, g, b = red*prob, green*prob, blue*prob
        print(f'<span style="background: rgba({r}, {g}, {b}, {alpha}">', end='', file=strio)
        print(tok, end='', file=strio)
        print('</span>', file=strio)
    strio.seek(0)
    toret = strio.read()
    return toret