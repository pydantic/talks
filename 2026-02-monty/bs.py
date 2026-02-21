import re
from dataclasses import dataclass, field
from typing import Any, cast

from bs4 import BeautifulSoup, Tag as BsTag


@dataclass
class Tag:
    """A mirror of a BeautifulSoup `Tag`."""

    name: str
    attrs: dict[str, str | list[str]] = field(default_factory=dict)
    string: str | None = None
    text: str = ''
    html: str = ''


def find(
    tag: Tag, name: str | None = None, attrs: dict[str, str] | None = None, string: str | None = None
) -> Tag | None:
    """Find the first descendant tag matching the criteria.

    Args:
        name: Tag name to match, e.g. `'a'`, `'div'`.
        attrs: Attribute key-value pairs to filter on.
        string: Match tags whose `.string` equals this value.

    Returns:
        The first matching `Tag`, or `None` if no match is found.
    """
    # bs4's types are horrible, this is the easiest work around
    result = _parse(tag.html).find(name, cast(Any, attrs), string=cast(Any, string))
    if result is None:
        return None
    else:
        return from_beautifulsoup(result)


def find_all(
    tag: Tag,
    name: str | re.Pattern[str] | None = None,
    attrs: dict[str, str] | None = None,
    string: str | None = None,
    limit: int | None = None,
) -> list[Tag]:
    """Find all descendant tags matching the criteria.

    Args:
        name: Tag name or compiled regex to match.
        attrs: Attribute key-value pairs to filter on.
        string: Match tags whose `.string` equals this value.
        limit: Stop after finding this many results.

    Returns:
        A list of matching `Tag` objects.
    """
    # bs4's types are horrible, this is the easiest work around
    results = _parse(tag.html).find_all(name, cast(Any, attrs), string=cast(Any, string), limit=limit)
    return [from_beautifulsoup(r) for r in results]


def select(tag: Tag, selector: str) -> list[Tag]:
    """Find all descendants matching a CSS selector.

    Args:
        selector: A CSS selector string, e.g. `'div.class > a'`.

    Returns:
        A list of matching `Tag` objects.
    """
    return [from_beautifulsoup(r) for r in _parse(tag.html).select(selector)]


def select_one(tag: Tag, selector: str) -> Tag | None:
    """Find the first descendant matching a CSS selector.

    Args:
        selector: A CSS selector string, e.g. `'div.class > a'`.

    Returns:
        The first matching `Tag`, or `None` if no match is found.
    """
    result = _parse(tag.html).select_one(selector)
    if result is None:
        return None
    return from_beautifulsoup(result)


def get(tag: Tag, key: str, default: str | None = None) -> str | list[str] | None:
    """Get an attribute value by key.

    Args:
        key: The attribute name, e.g. `'href'`, `'class'`.
        default: Value to return if the attribute is missing.

    Returns:
        The attribute value (a `str`, or `list[str]` for multi-valued
        attributes like `class`), or `default` if not found.
    """
    return tag.attrs.get(key, default)


def get_text(tag: Tag, separator: str = '', strip: bool = False) -> str:
    """Extract all text content within this tag.

    Args:
        separator: String inserted between text fragments.
        strip: Whether to strip whitespace from each fragment.

    Returns:
        The concatenated text content.
    """
    return _parse(tag.html).get_text(separator=separator, strip=strip)


def children(tag: Tag) -> list[Tag | str]:
    """Get the direct children of this tag.

    Returns:
        A list where each element is either a `Tag` or a `str`
        (for navigable text nodes).
    """
    out: list[Tag | str] = []
    for child in _parse(tag.html).children:
        if isinstance(child, BsTag):
            out.append(from_beautifulsoup(child))
        else:
            text = str(child)
            if text:
                out.append(text)
    return out


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def from_beautifulsoup(element: BsTag) -> Tag:
    """Convert a BeautifulSoup `Tag` into a Monty-compatible `Tag` dataclass."""
    assert isinstance(element, BsTag), f'Expected a BeautifulSoup Tag, got {type(element)}'
    string_val = element.string
    return Tag(
        name=element.name,
        attrs=dict(element.attrs),
        string=str(string_val) if string_val is not None else None,
        text=element.get_text(),
        html=str(element),
    )


def beautiful_soup(html: str) -> Tag:
    """Parse html with BeautifulSoup and return a `Tag`.

    Use this tool to get back a `Tag` object that can be used to extract information from HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    return from_beautifulsoup(soup)


tools = [
    beautiful_soup,
    find,
    find_all,
    select,
    select_one,
    get,
    get_text,
    children,
]


def _parse(html: str) -> BsTag:
    """Re-parse stored HTML into a BeautifulSoup tag.

    If the HTML represents a full document the `BeautifulSoup` object
    itself is returned (it behaves like a Tag).  Otherwise the first
    child tag is returned so that `find`/`select` operate on the
    correct element.
    """
    soup = BeautifulSoup(html, 'html.parser')
    # If the html was a single tag, unwrap so searches are scoped correctly.
    children = list(soup.children)
    if len(children) == 1 and isinstance(children[0], BsTag):
        return children[0]
    return soup
