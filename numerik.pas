unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, multiarray;

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
  //if Not KeepDims then
  //  SqueezeMultiArray(Result);
end;

end.

