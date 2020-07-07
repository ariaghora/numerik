unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, math, multiarray;

type
  TGenFunc = function(Params: array of single): single;

  { Wrappers of generator functions }
  function _Random(Params: array of single): single;
  function _RandG(Params: array of single): single;

  { Wrappers of single unary functions }
  function _DegToRad(X: Single; params: array of single): Single;
  function _Exp(X: Single; params: array of single): Single;
  function _Sin(X: Single; params: array of single): Single;

  { ----------- Trigonometry ----------- }
  function DegToRad(X: TMultiArray): TMultiArray; overload;
  function Sin(X: TMultiArray): TMultiArray; overload;

  { ----------- Exponential ----------- }
  function Exp(X: TMultiArray): TMultiArray; overload;

  { ----------- Random number generator ----------- }
  function Random(Shape: array of longint): TMultiArray; overload;
  function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

  function GenerateMultiArray(Shape: array of longint; GenFunc: TGenFunc;
    Params: array of single): TMultiArray;

implementation

  function _Random(Params: array of single): single;
  begin
    Exit(Random);
  end;

  function _RandG(Params: array of single): single;
  begin
    Exit(RandG(Params[0], Params[1]));
  end;

  function _DegToRad(X: Single; params: array of single): Single;
  begin
    Exit(DegToRad(X));
  end;

  function _Exp(X: Single; params: array of single): Single;
  begin
    Exit(Exp(X));
  end;

  function _Sin(X: Single; params: array of single): Single;
  begin
    Exit(Sin(X));
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

  function GenerateMultiArray(Shape: array of longint; GenFunc: TGenFunc;
    Params: array of single): TMultiArray;
  var
    i: longint;
  begin
    Result := AllocateMultiArray(specialize Prod<longint>(Shape));
    for i := 0 to High(Result.Data) do
      Result.Data[i] := GenFunc(Params);
    Result := Result.Reshape(Shape);
  end;

end.

