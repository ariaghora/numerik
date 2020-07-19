{
  A unit containing:

  @unorderedList(
    @item(Mathematical operation over TMultiArrays)
    @item(Some linear algebra functionality )
    @item(Random number generation )
  )
}

unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, multiarray, numerik.blas;

type
  TReduceFunc = function(A, B: TMultiArray): TMultiArray;

  { Holding the result of SVD }
  TSVDResult = record
    U, Sigma, VT: TMultiArray;
  end;

{ ----------- Arithmetic------------------------------------------------------ }

{ Compute the mean of a A along axis. The default axis is -1, meaning the mean is
  computed over all items in A. }
function Mean(A: TMultiArray; axis: integer = -1): TMultiArray; overload;

{ Compute the sum of a A along axis. The default axis is -1, meaning the sum is
  computed over all items in A. }
function Sum(A: TMultiArray; axis: integer = -1): TMultiArray; overload;

{ ----------- Trigonometry --------------------------------------------------- }

{ Compute element wise cosine over A }
function Cos(A: TMultiArray): TMultiArray; overload;
{ Convert degree to radian element wise over A }
function DegToRad(A: TMultiArray): TMultiArray; overload;
{ Compute element wise sine over A }
function Sin(A: TMultiArray): TMultiArray; overload;

{ ----------- Exponential ---------------------------------------------------- }

{ Compute element wise exponent over A }
function Exp(A: TMultiArray): TMultiArray; overload;

{ ----------- Random number generator ---------------------------------------- }

{ Wrapper for pascal's Random function, as a TMultiArray. }
function Random(Shape: array of longint): TMultiArray; overload;

{ Gaussian random array generator. }
function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

{ Reduce TMultiArray along a specified axis with specified ReduceFunc. This is
  for convenience rather than performance. }
function ReduceAlongAxis(A: TMultiArray; ReduceFunc: TReduceFunc; axis: integer;
     KeepDims: Boolean = False): TMultiArray;

{ Perform pseudoinverse over a matrix A }
function PseudoInverse(A: TMultiArray): TMultiArray;

{ Perform singular value decomposition (SVD) over a matrix A }
function SVD(A: TMultiArray; SigmasAsMatrix: Boolean=False): TSVDResult;


implementation

function _Cos(X: single; params: array of single): single;
begin
  Exit(System.Cos(X));
end;

function Cos(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Cos, []));
end;

function _DegToRad(X: single; params: array of single): single;
begin
  Exit(math.DegToRad(X));
end;

function DegToRad(A: TMultiArray): TMultiArray; overload;
begin
  Exit(ApplyUFunc(A, @_DegToRad, []));
end;

function _Exp(X: single; params: array of single): single;
begin
  Exit(System.Exp(X));
end;

function Exp(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Exp, []));
end;

function _Sin(X: single; params: array of single): single;
begin
  Exit(System.Sin(X));
end;

function Sin(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Sin, []));
end;

function _Random(Params: array of single): single;
begin
  Exit(System.Random);
end;

function Random(Shape: array of longint): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Random, []));
end;

function _RandG(Params: array of single): single;
begin
  Exit(math.RandG(Params[0], Params[1]));
end;

function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;
begin
  Exit(GenerateMultiArray(Shape, @_RandG, [mean, stddev]));
end;

function ReduceAlongAxis(A: TMultiArray; ReduceFunc: TReduceFunc; axis: integer;
     KeepDims: Boolean = False): TMultiArray;
var
  i: longint;
  IterIndices: array of TLongVector;
begin
  SetLength(IterIndices, Length(A.Indices));
  for i := 0 to High(IterIndices) do
    IterIndices[i] := []; // slicing by [ALL, ALL, ..., ALL]

  IterIndices[axis] := [0];
  Result := A.Slice(IterIndices);  // Slice([ALL, ..., [0], ..., ALL])

  for i := 1 to A.Shape[axis] - 1 do
  begin
    IterIndices[axis] := [i];
    Result := ReduceFunc(Result, A.Slice(IterIndices));//Result + A.Slice(IterIndices);
  end;

  if Not KeepDims then
    SqueezeMultiArrayAt(Result, axis);
end;

function Mean(A: TMultiArray; axis: integer = -1): TMultiArray; overload;
begin
  if axis = -1 then
    Exit(math.Mean(A.GetVirtualData));
  Exit(Sum(A, axis) / A.Shape[axis]);
end;

function Sum(A: TMultiArray; axis: integer = -1): TMultiArray;
begin
  if axis = -1 then
    Exit(math.Sum(A.GetVirtualData));
  Exit(ReduceAlongAxis(A, @Add, axis));
end;

function PseudoInverse(A: TMultiArray): TMultiArray;
var
  SVDRes: TSVDResult;
  i : integer;
begin
  SVDRes := SVD(A, True);
  for i := 0 to SVDRes.Sigma.Size - 1 do
    if SVDRes.Sigma.Get(i) > 0 then
       SVDRes.Sigma.Put(OffsetToIndex(SVDRes.Sigma, i), 1 / SVDRes.Sigma.Get(i));
  Exit(SVDRes.VT.T.Matmul(SVDRes.Sigma.T).Matmul(SVDRes.U.T));
end;

function SVD(A: TMultiArray; SigmasAsMatrix: Boolean=False): TSVDResult;
var
  i, m, n, info: integer;
  U, Sigma, VT: TMultiArray;
  Superb, SigmaVal: TSingleVector;
begin
  m := A.Shape[0];
  n := A.Shape[1];
  SetLength(Superb, min(m, n) - 1);
  SetLength(SigmaVal, min(m, n));

  U := AllocateMultiArray(m ** 2).Reshape([m, m]);
  VT := AllocateMultiArray(n ** 2).Reshape([n, n]);

  info := LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, CopyVector(A.GetVirtualData),
    n, SigmaVal, U.Data, m, VT.Data, n, Superb);

  Result.U := U;
  Result.VT := VT;

  if not SigmasAsMatrix then
    Result.Sigma := SigmaVal
  else
  begin
    Sigma := AllocateMultiArray(m * n).Reshape([m, n]);
    for i := 0 to min(m, n) - 1 do
      Sigma.Put([i, i], SigmaVal[i]);
    Result.Sigma := Sigma;
  end;

  superb := nil;
end;

end.

