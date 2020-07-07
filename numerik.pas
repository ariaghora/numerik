unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, multiarray;

{ ----------- Trigonometry ----------- }
function _Cos(X: single; params: array of single): single;
function _DegToRad(X: single; params: array of single): single;
function _Sin(X: single; params: array of single): single;
function Cos(X: TMultiArray): TMultiArray; overload;
function DegToRad(X: TMultiArray): TMultiArray; overload;
function Sin(X: TMultiArray): TMultiArray; overload;

{ ----------- Exponential ----------- }
function _Exp(X: single; params: array of single): single;
function Exp(X: TMultiArray): TMultiArray; overload;

{ ----------- Random number generator ----------- }
function _Random(Params: array of single): single;
function _RandG(Params: array of single): single;
function Random(Shape: array of longint): TMultiArray; overload;
function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

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

end.

