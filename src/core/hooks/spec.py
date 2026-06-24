# src/core/hooks/spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
import re
import torch

# ---------------------------------------------------------------------
# Public constants / helpers
# ---------------------------------------------------------------------
SEP = "@"
METHOD_SEP = "::"
BranchToken = Union[int, str]


def _parse_branch_token(token: str) -> BranchToken:
    token = token.strip()
    if token == "":
        raise ValueError("Branch token cannot be empty")
    if token.lstrip("-").isdigit():
        try:
            return int(token)
        except Exception:
            pass
    return token


def _collect_allowed_children(prefix: str, allowed: Optional[Iterable[str]]) -> List[str]:
    if allowed is None:
        return []
    prefix_sep = prefix + SEP
    children: set[str] = set()
    for candidate in allowed:
        if candidate == prefix:
            continue
        if candidate.startswith(prefix_sep):
            remainder = candidate[len(prefix_sep):]
            if not remainder:
                continue
            child = remainder.split(SEP, 1)[0]
            if child != "":
                children.add(child)
    return list(children)

def sanitize_id(s: str) -> str:
    """파일명/키로 쓰기 안전한 식별자."""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", s).strip("_")


def _split_attr_suffix(raw: str) -> Tuple[str, Optional[str]]:
    """
    Separate an optional '#attr' suffix from the spec core.
    Examples:
        'model.blocks.9::sae_layer#latent' -> ('model.blocks.9::sae_layer', 'latent')
        'model.blocks.9#pos_emb' -> ('model.blocks.9', 'pos_emb')
    """
    if "#" not in raw:
        return raw, None
    core, attr = raw.rsplit("#", 1)
    attr_clean = attr.strip()
    if attr_clean == "":
        raise ValueError(f"Attribute suffix '#{attr}' cannot be empty in spec '{raw}'.")
    return core.strip(), attr_clean

# ---------------------------------------------------------------------
# Tree walk / flatten
# ---------------------------------------------------------------------
def walk_tensors(
    base_name: str,
    output: Any,
    *,
    allowed_prefixes: Optional[Iterable[str]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """
    모델 출력 트리를 (이름, 텐서) 리스트로 평탄화.
    - list/tuple/dict 재귀
    - 텐서만 수집
    - allowed_prefixes 가 주어지면 해당 prefix (혹은 그 상/하위)만 탐색
    """
    out: List[Tuple[str, torch.Tensor]] = []
    allowed = tuple(allowed_prefixes) if allowed_prefixes is not None else None

    def _tokens(path: str) -> List[str]:
        return [tok for tok in path.split(SEP) if tok != ""]

    def _want_branch(prefix: str) -> bool:
        if allowed is None:
            return True
        prefix_tokens = _tokens(prefix)
        for a in allowed:
            allowed_tokens = _tokens(a)
            j = 0
            for tok in prefix_tokens:
                if j < len(allowed_tokens) and tok == allowed_tokens[j]:
                    j += 1
                    continue
                if tok.isdigit() and (j >= len(allowed_tokens) or not allowed_tokens[j].isdigit()):
                    continue
                break
            else:
                return True
        return False

    def _allow_leaf(prefix: str) -> bool:
        if allowed is None:
            return True
        prefix_tokens = _tokens(prefix)
        for a in allowed:
            allowed_tokens = _tokens(a)
            j = 0
            for tok in prefix_tokens:
                if j < len(allowed_tokens) and tok == allowed_tokens[j]:
                    j += 1
                    continue
                if tok.isdigit() and (j >= len(allowed_tokens) or not allowed_tokens[j].isdigit()):
                    continue
                break
            else:
                if j == len(allowed_tokens):
                    return True
        return False

    def _walk(prefix: str, obj: Any):
        if not _want_branch(prefix):
            return
        if torch.is_tensor(obj):
            if _allow_leaf(prefix):
                out.append((prefix, obj))
            return
        if isinstance(obj, (list, tuple)):
            for i, it in enumerate(obj):
                child = f"{prefix}{SEP}{i}"
                if _want_branch(child):
                    _walk(child, it)
        elif isinstance(obj, dict):
            for k, it in obj.items():
                child = f"{prefix}{SEP}{k}"
                if _want_branch(child):
                    _walk(child, it)
        else:
            for child in _collect_allowed_children(prefix, allowed):
                if child.isdigit():
                    # numeric child handled via list indexing already
                    continue
                try:
                    attr = getattr(obj, child)
                except AttributeError:
                    continue
                if callable(attr):
                    continue
                child_prefix = f"{prefix}{SEP}{child}"
                if _want_branch(child_prefix):
                    _walk(child_prefix, attr)

    _walk(base_name, output)
    return out

# ---------------------------------------------------------------------
# Legacy helper kept for compatibility
# ---------------------------------------------------------------------
def split_layer_and_branch(layer: str) -> Tuple[str, Optional[BranchToken]]:
    """
    Split a spec into base + branch using the *first* SEP.
    This preserves nested branch paths like 'module@key@0' as:
      base='module', branch='key@0'
    """
    if SEP in layer:
        base, idx = layer.split(SEP, 1)
        try:
            return base, int(idx)
        except ValueError:
            if idx.strip() == "":
                return base, None
            return base, idx
    return layer, None

# ---------------------------------------------------------------------
# Spec parsing with alias support
# ---------------------------------------------------------------------
def parse_spec(spec: str) -> ParsedSpec:
    """
    Return: ParsedSpec (iterable like (base, method, branch, alias, attr) for backwards compatibility)

    Accepted forms (공백 허용, 괄호는 무시됨):
      # 모듈 forward
      "backbone.stage3"                -> base=..., method=None, branch=None
      "backbone.stage3@2"              -> base, None, branch=2
      "model.patch_embed@0"            -> base=model.patch_embed, branch=0
      "model@all_frame_outputs"        -> base=model, branch="all_frame_outputs"
      # 임의 메서드 반환
      "enc.pos::forward_with_coords"   -> base=enc.pos, method=forward_with_coords
      "enc.pos::forward_with_coords@0" -> method 분기 0
      "enc.pos@0::foo@latent"          -> base=enc.pos, branch=0 (base), method foo, branch='latent' (method)
      # 별칭 (둘 다 지원)
      "alias=enc.pos::forward_with_coords@0"
      "enc.pos::forward_with_coords@0 => alias"
      "enc.pos::forward_with_coords@0 as alias"

    규칙:
      - '::' 있으면 우변은 메서드 이름(괄호 무시), 브랜치는 메서드 쪽에 붙임
      - '::' 없으면 브랜치는 base 쪽
      - alias 는 prefix('alias=...') 또는 suffix('... => alias' / '... as alias')
      - 반환 alias 는 sanitize_id 로 정리하지 않음(원문 보존). 파일명/키로 쓸 땐 sanitize_id 사용.
      # 속성 선택
      "model.blocks.9::sae_layer#latent" -> attr='latent'
      "model.blocks.9#current_pos_emb"   -> attr='current_pos_emb'
    The returned dataclass exposes `.base`, `.base_branch`, `.method`, `.method_branch`,
    `.branch` (legacy behaviour), `.alias`, `.attr`, and helpers `.base_with_branch`
    / `.method_with_branch`.
    """
    s = spec.strip()

    alias: Optional[str] = None
    # suffix alias: => alias  (우선 처리)
    if "=>" in s:
        left, right = s.rsplit("=>", 1)
        s, alias = left.strip(), right.strip()
    else:
        # suffix alias:  as alias
        m = re.search(r"\s+as\s+(.+)$", s)
        if m:
            s, alias = s[:m.start()].strip(), m.group(1).strip()
        else:
            # prefix alias:  alias=
            if "=" in s and not s.strip().startswith(("http://", "https://")):
                # 가장 앞쪽 '=' 를 alias 구분자로 사용
                a, rest = s.split("=", 1)
                # '=' 가 경로 일부일 가능성 거의 없음(경로엔 '=' 잘 안 씀)
                if a and rest:
                    alias, s = a.strip(), rest.strip()

    # attribute suffix
    s, attr = _split_attr_suffix(s)

    # 메서드/브랜치 파싱
    if METHOD_SEP in s:
        base, rhs = s.split("::", 1)
        # 괄호 제거
        if rhs.endswith("()"):
            rhs = rhs[:-2]
        branch: Optional[BranchToken] = None
        method_part = rhs.strip()
        if SEP in method_part:
            method_raw, b = method_part.split(SEP, 1)
            method = method_raw.strip()
            if b.strip():
                branch = _parse_branch_token(b)
        else:
            method = method_part
        base_path, base_branch = split_layer_and_branch(base.strip())
        if base_branch is not None and branch is None:
            branch = base_branch
        return ParsedSpec(
            base=base_path.strip(),
            method=method.strip(),
            base_branch=base_branch,
            method_branch=branch,
            alias=(alias.strip() if alias else None),
            attr=attr,
        )
    else:
        base = s
        branch: Optional[BranchToken] = None
        if SEP in s:
            base, b = s.split(SEP, 1)
            if b.strip():
                branch = _parse_branch_token(b)
        return ParsedSpec(
            base=base.strip(),
            method=None,
            base_branch=branch,
            method_branch=None,
            alias=(alias.strip() if alias else None),
            attr=attr,
        )

def canonical_name(
    base: str,
    method: Optional[str],
    branch: Optional[BranchToken],
    alias: Optional[str],
    attr: Optional[str] = None,
) -> str:
    """사람/머신이 동일하게 인식할 수 있는 정규화된 이름."""
    if alias:
        return alias
    prefix = f"{base}{METHOD_SEP}{method}" if method else base
    if branch is not None:
        prefix = f"{prefix}{SEP}{branch}"
    if attr:
        return f"{prefix}#{attr}"
    return prefix
@dataclass(frozen=True)
class ParsedSpec:
    base: str
    method: Optional[str]
    base_branch: Optional[BranchToken]
    method_branch: Optional[BranchToken]
    alias: Optional[str]
    attr: Optional[str]

    @property
    def branch(self) -> Optional[BranchToken]:
        return self.method_branch if self.method_branch is not None else self.base_branch

    @property
    def base_with_branch(self) -> str:
        if self.base_branch is None:
            return self.base
        return f"{self.base}{SEP}{self.base_branch}"

    @property
    def method_with_branch(self) -> Optional[str]:
        if self.method is None:
            return None
        if self.method_branch is None:
            return self.method
        return f"{self.method}{SEP}{self.method_branch}"

    def legacy_tuple(self) -> Tuple[str, Optional[str], Optional[BranchToken], Optional[str], Optional[str]]:
        return (self.base, self.method, self.branch, self.alias, self.attr)

    def __iter__(self):
        return iter(self.legacy_tuple())

    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int):
        return self.legacy_tuple()[idx]
