"""Image-text matching helpers backed by the Lite CLIPITM service."""

from vlm.itm.clipitm import CLIPITMClient


itmclient = CLIPITMClient(port=12182)


def get_itm_message(rgb_image, label, return_stats=False):
    response = itmclient.infer(rgb_image, f"Is there a {label} in the image?")
    result = (float(response["response"]), float(response["itm score"]))
    if return_stats:
        return result + (response.get("timing", {}),)
    return result


def get_itm_message_cosine(rgb_image, label, room, return_stats=False):
    if room != "everywhere":
        prompt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        prompt = f"Seems like there is a {label} ahead?"
    response = itmclient.infer(rgb_image, prompt)
    cosine = float(response["response"])
    if return_stats:
        return cosine, response.get("timing", {})
    return cosine
