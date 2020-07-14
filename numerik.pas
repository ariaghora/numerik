unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, multiarray;

const
  LIB_NAME = 'libopenblas.dll';

type
  CBLAS_ORDER     = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113);

{ ----------- Trigonometry --------------------------------------------------- }
function _Cos(X: single; params: array of single): single;
function _DegToRad(X: single; params: array of single): single;
function _Sin(X: single; params: array of single): single;
function Cos(X: TMultiArray): TMultiArray; overload;
function DegToRad(X: TMultiArray): TMultiArray; overload;
function Sin(X: TMultiArray): TMultiArray; overload;

{ ----------- Exponential ---------------------------------------------------- }
function _Exp(X: single; params: array of single): single;
function Exp(X: TMultiArray): TMultiArray; overload;

{ ----------- Random number generator ---------------------------------------- }
function _Random(Params: array of single): single;
function _RandG(Params: array of single): single;

{ Wrapper for pascal's Random function, as a TMultiArray. }
function Random(Shape: array of longint): TMultiArray; overload;

{ Gaussian random array generator. }
function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

{ Sum-reducing TMultiArray along a specified axis. This is for convenience rather
  than performance. }
function ReduceSum(A: TMultiArray; axis: integer; KeepDims: Boolean = False): TMultiArray;

{ BLAS backend interface }
procedure cblas_sgemm(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: single; A: TSingleVector; lda: longint; B: TSingleVector;
    ldb: longint; beta: single; C: TSingleVector; ldc: longint);
    external LIB_NAME;

procedure LAPACKE_sgesvd(JOBU, JOBVT: char; M, N: longint; A: TSingleVector;
    LDA: longint; S, U: TSingleVector; LDU: longint; VT: TSingleVector;
    LDVT: longint; Work: TSingleVector; LWork, Info: longint);
    external LIB_NAME;

function MatMul_BLAS(A, B: TMultiArray): TMultiArray;


implementation

function _Random(Params: array of single): single;
begin
  Exit(Random);
end;

function _RandG(Params: array of single): single;
begin
  Exit(RandG(Params[0], Params[1]));
end;

function _Cos(X: single; params: array of single): single;
begin
  Exit(Cos(X));
end;

function _DegToRad(X: single; params: array of single): single;
begin
  Exit(DegToRad(X));
end;

function _Exp(X: single; params: array of single): single;
begin
  Exit(Exp(X));
end;

function _Sin(X: single; params: array of single): single;
begin
  Exit(Sin(X));
end;

function Cos(X: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(X, @_Cos, []));
end;

function DegToRad(X: TMultiArray): TMultiArray; overload;
begin
  Exit(ApplyUFunc(X, @_DegToRad, []));
end;

function Sin(X: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(X, @_Sin, []));
end;

function Exp(X: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(X, @_Exp, []));
end;

function Random(Shape: array of longint): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Random, []));
end;

function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;
begin
  Exit(GenerateMultiArray(Shape, @_RandG, [mean, stddev]));
end;

function ReduceSum(A: TMultiArray; axis: integer; KeepDims: Boolean = False): TMultiArray;
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
    Result := Result + A.Slice(IterIndices);
  end;

  if Not KeepDims then
    SqueezeMultiArray(Result);
end;

function MatMul_BLAS(A, B: TMultiArray): TMultiArray;
var
  OutSize: longint;
begin
  OutSize := A.Shape[0] * B.Shape[1];
  Result := AllocateMultiArray(OutSize).Reshape([A.Shape[0], B.Shape[1]]);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              A.Shape[0], B.shape[1], B.shape[0], // A.shape[0], B.shape[1], B.shape[0]
              1,                                  // alpha
              A.GetVirtualData, B.Shape[0],       // A, B.shape[0]
              B.GetVirtualData, B.Shape[1],       // B, B.shape[1]
              1,                                  // beta
              Result.Data, B.Shape[1]             // C, B.shape[1]
              );
end;

end.

