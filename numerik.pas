unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, math, multiarray;

  { Wrappers of single functions }
  function _DegToRad(X: Single): Single;
  function _Exp(X: Single): Single;
  function _Sin(X: Single): Single;

  { ----------- Trigonometry ----------- }
  function DegToRad(X: TMultiArray): TMultiArray; overload;
  function Sin(X: TMultiArray): TMultiArray; overload;

  { ----------- Exponential ----------- }
  function Exp(X: TMultiArray): TMultiArray; overload;

  { ----------- Random number generator ----------- }
  function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

implementation

  function _DegToRad(X: Single): Single;
  begin
    Exit(DegToRad(X));
  end;

  function _Exp(X: Single): Single;
  begin
    Exit(Exp(X));
  end;

  function _Sin(X: Single): Single;
  begin
    Exit(Sin(X));
  end;

  function DegToRad(X: TMultiArray): TMultiArray; overload;
  begin
    Exit(ApplyUFunc(X, @_DegToRad));
  end;

  function Sin(X: TMultiArray): TMultiArray;
  begin
    Exit(ApplyUFunc(X, @_Sin));
  end;

  function Exp(X: TMultiArray): TMultiArray;
  begin
    Exit(ApplyUFunc(X, @_Exp));
  end;

  function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;
  var
    i: longint;
  begin
    Result := AllocateMultiArray(specialize Prod<longint>(Shape));
    for i := 0 to High(Result.Data) do
      Result.Data[i] := RandG(mean, stddev);
    Result := Result.Reshape(Shape);
  end;

end.

